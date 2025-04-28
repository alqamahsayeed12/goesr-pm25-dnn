#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:25:27 2025

@author: asayeed
"""

# src/pipeline.py
from src.config import PATHS, CONSTANTS
from src.utils import normalize, moving_average_with_nan_2D, custom_loss1
from src.preprocess import merge_GOES, szafunc
import keras
import xarray as xr
import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl


#%% Setting Global Varaible & Parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
projection=ccrs.PlateCarree()

cbarticks = list(np.arange(0, 121,20))#+list(np.arange(200, 501,50))+[1000]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["darkgreen","yellow","darkred",'maroon'])
cmaplist = [cmap(i) for i in range(cmap.N)][50:]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
norm = mpl.colors.BoundaryNorm(cbarticks, cmap.N) 

###-- Dictionary to define datatype of each parameter. to be called if required
DTYPE  = {'time':"int",
          "projection_y_coordinate":'int',
          "projection_x_coordinate":"int",
          "Date_UTC":"str",
          "RH": "float32",
          "MAXUW_1hr_max_fcst":"float32",
          "MAXVW_1hr_max_fcst":"float32",
          "TMP":"float32"}

#%%
def run_pipeline(goes_fn, collocated, hrrr_vars, var_index, feature_columns, feature_columns2, mx, mn, dnn_cols, plot=False):
    OUTPUT_NC_FILENAME = goes_fn[-32:-3]+"_dnn_v1_4.nc"
    OUTPUT_NC_Folder = str(PATHS['output_nc']) + "/"
    
    ## Get date and hour from filename
    date = goes_fn[-13:-5]
    hour = (goes_fn[-5:-3])
    
    ## Convert to datetime
    date = pd.to_datetime(date+hour,format="%Y%m%d%H")
    

    if not os.path.exists(OUTPUT_NC_Folder+OUTPUT_NC_FILENAME):
       
        try:


            ###--- Merge GOES east and west
            goes_ = merge_GOES(goes_fn)           
            
            ###--- Extract pm from goes dataset
            pm = goes_["pm25sat_com"].values
            
            ###--- Create 3 x 3 grid average of PM2.5
            pm_avg = moving_average_with_nan_2D(pm,3)
            
            ###--- Add new variable to existing dataset 
            goes_["pm_avg"]=(["ydim","xdim"],  pm_avg.astype("float32"))
            
            ###--- Similar processing for aod
            aod = goes_["abi_aod"].values
            aod_avg = moving_average_with_nan_2D(aod,3)
            goes_["aod_avg"]=(["ydim","xdim"],  aod_avg.astype("float32"))
            
            ###--- Convert GOES dataset to pandas like dataframe
            goes=goes_.to_dataframe().reset_index()
            
            ###--- Add date column to dataframe
            goes["Date_UTC"]=date
            
            ###--- Remove missing pixels from the newly generated avgreges 
            goes.pm_avg[np.isnan(goes.lon_com)]=np.nan
            goes.aod_avg[np.isnan(goes.lon_com)]=np.nan
            
            ###--- Free up RAM by removing excess variables
            del pm,pm_avg,aod,aod_avg
            gc.collect()
            
            ###--- Get correct dates for HRRR data download
            if int(hour)<12:
                date2 = (date - pd.to_timedelta(12,unit="H")).strftime("%Y%m%d")
            else:
                date2 = (date).strftime("%Y%m%d")
            
            
            hrrr_fname = date2 + "_12z_df.csv" ### We are using model initilaized 12 UTC

            hrrr_file_path = str(PATHS['hrrr_folder']) + "/" + hrrr_fname

            if not os.path.exists(hrrr_file_path):
                
                print("HRRR file missing: Downloading.")
                
                ###--- Open Base Dataframe for HRRR
                ###--- base_df have collocated drid points with HRRR            
                hrrr = pd.read_csv( str(PATHS['scalar_folder']) + "/base_df.csv",index_col=0)
                
                #%% ###--- Only Download the required variables
                for i in var_index:                      
                    row = hrrr_vars.iloc[i,:]
                    level = row["ZARR_LEVEL_NAME"]
                    var = row["Parameter Short Name"]   
                    
                    ###--- Define url
                    url = f"s3://hrrrzarr/sfc/{date2}/{date2}_12z_fcst.zarr/{level}/{var}/{level}/"                   
                 
                    
                    ###--- Download if varibale file is avaialbe on AWS-S3         
                    try:
                        
                        ds = xr.open_dataset(url, engine="zarr", 
                                             backend_kwargs={"storage_options": {"anon": True}}, consolidated=False)

                        a = ds.to_dataframe().reset_index()
                        a = a[a.time<24]
                        hrrr = hrrr.merge(a,how="inner")
                        del a,ds
                        gc.collect()              
                    except Exception as e:                
                        print (str(e))
                
                #%% ###--- Add date and time to hrrr dataframe
                hrrr["Date_UTC"] = pd.to_datetime(date2)
                hrrr["Date_UTC"] = hrrr["Date_UTC"] +pd.to_timedelta(hrrr.time.values+12,unit="H")
                
                ###--- Save the HRRR file
                hrrr.to_csv(hrrr_file_path)#process hrrr
                print ("Downloading HRRR Done...")
                return
            else:
                print("HRRR file found: Skip Downloading.")
                

            hrrr = pd.read_csv(hrrr_file_path, parse_dates=True, index_col=0)
            
            ###--- Preprocess HRRR dataframe
            hrrr['Date_UTC'] = pd.to_datetime(hrrr.Date_UTC,format="mixed")
            hrrr = hrrr[hrrr.Date_UTC==date]
            hrrr = hrrr.rename(columns={'projection_y_coordinate':"hrrr_i",
                                      'projection_x_coordinate':"hrrr_j"})
            hrrr["Wind Speed"] = (hrrr[ 'MAXUW_1hr_max_fcst']**2 + hrrr['MAXVW_1hr_max_fcst']**2)**0.5
            
            ###--- Rename goes dimensions columns

            goes = goes.rename(columns={'ydim':"goes_i",
                                      'xdim':"goes_j"})
            
            ###--- Create dataframe by merging goes on collocation dataframe
            DF = pd.merge(collocated,goes,how="inner",on=["goes_i","goes_j"])
            
            ###--- Merege HRRR on earlier created dataframe
            ###--- This will only merge collocated grids
            DF = pd.merge(DF,hrrr,how="inner",on=["hrrr_i","hrrr_j","Date_UTC"])
            
            ###--- Find the index of 
            idx = np.where(np.isnan(collocated.hrrr_i))[0]
            
            DF = DF.drop(['lon_com_x', 'lat_com_x', 
                   'hrrr_lat', 'hrrr_lon', 'time'],axis=1)
            
            
        
            ###--- Saving extra grid points as seperate dataframe
            DF2 = goes.iloc[idx,:]
            
            DF2 = DF2.rename(columns={"lat_com":"Lat",
                                    "lon_com":"Lon" ,
                                    'smoke_dust_mask':'smoke_dust_mask_ge', 
                                    'abi_aod':'abi_aod_ge'})
            
            del hrrr,goes
            gc.collect()
            
            ###--- Calculate SED
            DF['SED'] = 1 - 0.01672 * np.cos(np.radians(0.9856 * (np.array((date.strftime('%j')),dtype='float')-4))) 
            
            
            ###--- Rename columns
            DF = DF.rename(columns={"lat_com_y":"Lat",
                                    "lon_com_y":"Lon" ,
                                    'smoke_dust_mask':'smoke_dust_mask_ge', 
                                    'abi_aod':'abi_aod_ge'
                                    })
            ###--- Calculate SZA using parallization
            DF['SZA'] = szafunc(date, DF['Lat'], DF['Lon'])
            

            for kk in range(len(CONSTANTS['idxI'])):
                i, j = CONSTANTS['idxI'][kk], CONSTANTS['idxJ'][kk]
                try:
                    fn = str(i).zfill(4)+"_"+str(j).zfill(4)
                    model_path = str(PATHS['model_folder']) + fn + "_relu_best_dnn1.h5"
                    model_loaded = keras.models.load_model(model_path, custom_objects={'customLoss1': custom_loss1})
                except:
                    model_loaded = keras.models.load_model(str(PATHS['model_folder'])+"0000_0720_relu_best_dnn1.h5", custom_objects={'customLoss1': custom_loss1})
                DF['DNN_'+str(kk).zfill(2)] = model_loaded.predict(normalize(DF[feature_columns], mx[feature_columns], mn[feature_columns]), batch_size=4096, verbose=0)

            #### Load Ensemble model
            ensemble_model = keras.models.load_model(str(PATHS['model_folder'])+'_dnn_ensemble_v1_4.h5', custom_objects={'customLoss1': custom_loss1})
            DF['DNN_en'] = ensemble_model.predict(normalize(DF[feature_columns2], mx[feature_columns2], mn[feature_columns2]), batch_size=4096, verbose=0)

            ### Apply mask to remove extra pixel due to averaging kerenel            
            DF['Mask'] = DF["pm25sat_com"]//DF["pm25sat_com"]
            DF["DNN_en"] = DF["DNN_en"]*DF['Mask']

            ###--- add the previously removed rows
            DF = pd.concat((DF,DF2),axis=0)
            ###--- Remove averaged columns
            DF = DF.drop(["Date_UTC",'pm_avg', 'aod_avg', 'hrrr_i', 
                          'hrrr_j','Date_UTC', 'MAXUW_1hr_max_fcst',
                          'MAXVW_1hr_max_fcst', 'RH', 'TMP', 'Wind Speed', 'SED', 
                          'SZA','Mask']+dnn_cols,
                         axis=1)
            ###--- Set index as in original netCDF      
            out_df = DF.rename(columns={'Lat': 'lat', 'Lon': 'lon',"goes_i":'ydim',
                                      "goes_j":'xdim',"DNN_en":"pm25gwr_dnn_com_ensemble"}).set_index(['ydim', 'xdim'])
    
            del DF,DF2
            gc.collect()
            
            ###--- Create Dataset from dataframe
            out_xarray = xr.Dataset.from_dataframe(out_df)  
            out_xarray = out_xarray.astype("float32")
            
            ###--- Assign Metadata
            
            out_xarray=out_xarray.assign_coords({"lon_com":out_xarray.lon,"lat_com":out_xarray.lat})
            del out_xarray["ydim"],out_xarray["xdim"],out_xarray["lat"],out_xarray["lon"]

            variables = list(out_xarray.keys())
            for para in variables:
                # out_xarray[para]=(["ydim","xdim"],  var)
                out_xarray[para].encoding=goes_["pm25sat_com"].encoding
                out_xarray[para].encoding['coordinates']='lon lat'
                
            
            out_xarray.coords['lat_com'].attrs["long_name"] = 'latitude of GOES east and west combined'
            out_xarray.coords['lon_com'].attrs["long_name"] = 'longitude of GOES east and west combined'
            
            out_xarray.coords['lat_com'].attrs["units"] = 'degree_north'
            out_xarray.coords['lon_com'].attrs["units"] = 'degree_east'

            ###--- Save the dataset as netCDF file
            out_xarray.to_netcdf(OUTPUT_NC_Folder+OUTPUT_NC_FILENAME, mode='w', format='NETCDF4',
                                 encoding = {
                                      'lat_com': {"_FillValue":None,'zlib': True, 'dtype':'float32'},
                                              'lon_com': {"_FillValue":None,'zlib': True,'dtype':'float32'},
                                              'pm25sat_com':{"_FillValue":-999,'zlib': True,'dtype':'float32'},
                                              'pm25gwr_dnn_com_ensemble':{"_FillValue":-999,'zlib': True,'dtype':'float32'},
                                             })
            
            
            print(f"Saved output: {OUTPUT_NC_FILENAME}")


        except Exception as e:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_tb.tb_lineno)
            print(e)
            
    if plot:
        out_xarray = xr.open_dataset(OUTPUT_NC_Folder+OUTPUT_NC_FILENAME)
        lats = out_xarray['lat_com'].values
        lons = out_xarray['lon_com'].values
        for var, title in zip(['pm25sat_com', 'pm25gwr_dnn_com_ensemble'], ['GWR', 'DNN']):
            fig, ax = plt.subplots( 1,1, figsize=(30,20),subplot_kw={'projection': projection},frameon=True)
            ax.set_extent([-125, -65, 22, 50], crs=ccrs.PlateCarree())
            pcm = ax.pcolor(lons, lats, out_xarray[var].values, cmap=cmap, vmin=0, vmax=120,transform=ccrs.PlateCarree())
            ax.coastlines(resolution='50m')
            states_provinces = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='50m',
                    facecolor='none')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(states_provinces)
            
            plt.title(f'{title} PM2.5 {date.strftime("%Y-%m-%d %H:%M")}', fontsize=25,fontweight="bold")
            
            # ax.set_title('New', fontsize=50,fontweight="bold")
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cbar = plt.colorbar(pcm, cax=cax, ticks=cbarticks,extend='max', label='µg/m³')
            cbar.ax.tick_params(labelsize=14)
            plot_file = PATHS['plots'] / f'{title}_{date.strftime("%Y%m%d%H")}.png'
            plt.savefig(plot_file, bbox_inches='tight')           
            plt.close()
