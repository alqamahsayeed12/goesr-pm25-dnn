#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:18:09 2025

@author: asayeed
"""

# src/preprocess.py
import xarray as xr
import numpy as np
from .utils import moving_average_with_nan_2D

###--- 2.5. Helper function to merge GOES Data : Adding addtional rows to GOES-E and GOES-W 
def merge_GOES_helper(var_e,var_w,e,w,para):
    if ((para != "lat") and (para != "lon")):
        try:
            var_e[e]=np.nan
            var_w[w]=np.nan
        except: pass
    add_rows_e = np.zeros((122,2500))*np.nan
    var_e = np.concatenate((add_rows_e,var_e),axis=0)
    
    add_rows_w = np.zeros((422,1800))*np.nan
    var_w = np.concatenate((var_w,add_rows_w),axis=0)
    
    var = np.concatenate((var_w[:,:1254],var_e[:,367:]),axis=1)
    var[1200:,887:1254] = var[1200:,:367]
    var = var.astype("float32")
    return var
    

###--- 2.6 Merge GOES-E&W
def merge_GOES(f='file_name'):
    s = xr.open_dataset(f,decode_coords="all")
    
    pm_e = s['pm25sat_ge'].values
    pm_w = s['pm25sat_gw'].values
    
    lat_E = s['lat_ge'].values
    lat_E[lat_E == -999.] = np.nan
    
    lat_W = s['lat_gw'].values
    lat_W[lat_W == -999.] = np.nan
    
    lon_E = s['lon_ge'].values
    lon_E[lon_E == -999.] = np.nan
    
    lon_W = s['lon_gw'].values
    lon_W[lon_W == -999.] = np.nan
    
    e=np.where(lon_E <= -106.)
    w=np.where (lon_W >= -106.)
    
    #Create Merged lat, lon and pm
    lat = merge_GOES_helper(lat_E, lat_W,e,w,"lat")
    lon = merge_GOES_helper(lon_E, lon_W,e,w,"lon")
    pm = merge_GOES_helper(pm_e,pm_w,e,w,"pm")
   
        
    lat = lat.astype("float32")
    lon = lon.astype("float32")
    pm  = pm.astype("float32")
    
    #Create new dataset with merged grid with attributes and metadata from original file
    ds  = xr.Dataset(
        data_vars=dict(
            pm25sat_com=(["ydim","xdim"],pm),
            ),
        coords=dict(
            lon_com=(["ydim","xdim"],lon),
            lat_com= (["ydim","xdim"],lat)),
        attrs=s['pm25sat_ge'].attrs,
            )
        
    #Add metadata to new dataset
    ds["pm25sat_com"].encoding=s["pm25sat_ge"].encoding
    ds["pm25sat_com"].encoding['coordinates']='lon lat'
    ds["lat_com"].encoding=s["lat_ge"].encoding
    ds["lon_com"].encoding=s["lon_ge"].encoding
    
    ## Merge other parameters
    parameter = ['smoke_dust_mask', 'abi_aod', 'slope', 'intercept','count_aod','count_cm']
    for para in parameter:
        var_e = s[para+'_ge'].values
        var_w = s[para+'_gw'].values
        var = merge_GOES_helper(var_e, var_w, e, w,para)
        ds[para]=(["ydim","xdim"],  var)
        ds[para].encoding=s[para+'_ge'].encoding
        ds[para].encoding['coordinates']='lon lat'
       
    
    ds.coords['lat_com'].attrs["long_name"] = 'latitude of GOES'
    ds.coords['lon_com'].attrs["long_name"] = 'longitude of GOES'
    
    ds.coords['lat_com'].attrs["units"] = 'degree_north'
    ds.coords['lon_com'].attrs["units"] = 'degree_east'
    return ds

#%%
def szafunc(day, dLongitude, dLatitude):
    """
        inputs: day: datetime object
                dLongitude: longitudes (scalar or Numpy array)
                dLatitude: latitudes (scalar or Numpy array)
        output: solar zenith angles
    """
    dHours, dMinutes, dSeconds = day.hour, day.minute, day.second
    iYear, iMonth, iDay = day.year, day.month, day.day

    dEarthMeanRadius = 6371.01
    dAstronomicalUnit = 149597890

    ###################################################################
    # Calculate difference in days between the current Julian Day
    # and JD 2451545.0, which is noon 1 January 2000 Universal Time
    ###################################################################
    # Calculate time of the day in UT decimal hours
    dDecimalHours = dHours + (dMinutes + dSeconds / 60.) / 60.
    # Calculate current Julian Day
    liAux1 = int((iMonth - 14.) / 12.)
    liAux2 = int((1461. * (iYear + 4800. + liAux1)) / 4.) + int((367. * (iMonth - 2. - 12. * liAux1)) / 12.) - int((3. * int((iYear + 4900. + liAux1) / 100.)) / 4.) + iDay - 32075.
    dJulianDate = liAux2 - 0.5 + dDecimalHours / 24.
    # Calculate difference between current Julian Day and JD 2451545.0
    dElapsedJulianDays = dJulianDate - 2451545.0

    ###################################################################
    # Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
    # ecliptic in radians but without limiting the angle to be less than 2*Pi
    # (i.e., the result may be greater than 2*Pi)
    ###################################################################
    dOmega = 2.1429 - 0.0010394594 * dElapsedJulianDays
    dMeanLongitude = 4.8950630 + 0.017202791698 * dElapsedJulianDays  # Radians
    dMeanAnomaly = 6.2400600 + 0.0172019699 * dElapsedJulianDays
    dEclipticLongitude = dMeanLongitude + 0.03341607 * np.sin(dMeanAnomaly) + 0.00034894 * np.sin(2. * dMeanAnomaly) - 0.0001134 - 0.0000203 * np.sin(dOmega)
    dEclipticObliquity = 0.4090928 - 6.2140e-9 * dElapsedJulianDays + 0.0000396 * np.cos(dOmega)

    ###################################################################
    # Calculate celestial coordinates ( right ascension and declination ) in radians
    # but without limiting the angle to be less than 2*Pi (i.e., the result may be
    # greater than 2*Pi)
    ###################################################################
    dSin_EclipticLongitude = np.sin(dEclipticLongitude)
    dY = np.cos(dEclipticObliquity) * dSin_EclipticLongitude
    dX = np.cos(dEclipticLongitude)
    dRightAscension = np.arctan2(dY, dX)
    if dRightAscension < 0.0:
        dRightAscension = dRightAscension + 2.0 * np.pi
    dDeclination = np.arcsin(np.sin(dEclipticObliquity) * dSin_EclipticLongitude)

    ###################################################################
    # Calculate local coordinates ( azimuth and zenith angle ) in degrees
    ###################################################################
    dGreenwichMeanSiderealTime = 6.6974243242 + 0.0657098283 * dElapsedJulianDays + dDecimalHours
    dLocalMeanSiderealTime = (dGreenwichMeanSiderealTime * 15. + dLongitude) * (np.pi / 180.)
    dHourAngle = dLocalMeanSiderealTime - dRightAscension
    dLatitudeInRadians = dLatitude * (np.pi / 180.)
    
    dCos_Latitude = np.cos(dLatitudeInRadians)
    dSin_Latitude = np.sin(dLatitudeInRadians)
    dCos_HourAngle = np.cos(dHourAngle)
                
    dZenithAngle = (np.arccos(dCos_Latitude * dCos_HourAngle * np.cos(dDeclination) + np.sin(dDeclination) * dSin_Latitude))
    # dY = -np.sin(dHourAngle)
    # dX = np.tan(dDeclination) * dCos_Latitude - dSin_Latitude * dCos_HourAngle
    # dAzimuth = np.arctan2(dY, dX)
    # dAzimuth[dAzimuth < 0.0] = dAzimuth[dAzimuth < 0.0] + 2.0 * np.pi
    # dAzimuth = dAzimuth / (np.pi / 180.)
    # # Parallax Correction
    dParallax = (dEarthMeanRadius / dAstronomicalUnit) * np.sin(dZenithAngle)
    dZenithAngle = (dZenithAngle + dParallax) / (np.pi / 180.)
    
    # return dAzimuth - 180.0, dZenithAngle

    return dZenithAngle

# src/model.py
from keras.models import load_model
from .utils import custom_loss1

def predict_with_model(model_path, df, features):
    model = load_model(model_path, custom_objects={'customLoss1': custom_loss1})
    predictions = model.predict(df[features], batch_size=4096, verbose=0)
    return predictions
