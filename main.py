#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:54:14 2025

@author: asayeed
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run GOES PM2.5 estimation pipeline with command-line arguments, optional plotting, progress bar, and timing.
Created on Mon Apr 28 10:21:10 2025
Author: asayeed
"""

# Import necessary modules
from src import download, preprocess, config
from src.pipeline import run_pipeline
import glob
import pandas as pd
import os
import warnings
import argparse
from tqdm import tqdm
import time

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GOES PM2.5 Estimation Pipeline")
    parser.add_argument('--start', type=str, required=True, help='Start date in YYYYMMDD format')
    parser.add_argument('--end', type=str, required=True, help='End date in YYYYMMDD format')
    parser.add_argument('--plot', type=str, default='False', help='Enable plotting of results (True/False)')
    args = parser.parse_args()

    # Load paths from configuration
    from src.config import PATHS

    start = args.start
    end = args.end
    plot = args.plot.lower() == 'true'  # Parse 'True'/'False' string to boolean

    # Start timer
    total_start_time = time.time()

    # Download GOES files for specified dates
    download.download_goes(start, end)

    # Find downloaded GOES NetCDF files
    file_list = sorted(glob.glob(str(PATHS['goes_folder'] / f'*/pm25_gwr_aod_exp50_{start}*')))

    # Load collocation file between HRRR and GOES grids
    collocated = pd.read_csv(str(PATHS['scalar_folder'])+"/collocated_hrrr_on_goes.csv", index_col=0)

    # Load HRRR variable list
    hrrr_vars = pd.read_csv(str(PATHS['scalar_folder'])+"/hrrr_varaible_list_selected4.csv", index_col=0)

    # Indices of selected HRRR variables
    var_index = [36, 37, 45, 51]

    # Define feature columns for DNN models
    feature_columns = ['Lat', 'Lon', 'SED', 'SZA', 'Wind Speed', 'RH', 'TMP', 'smoke_dust_mask_ge', 'aod_avg', 'pm_avg']

    # Define DNN model output columns
    dnn_cols = [f'DNN_{str(i).zfill(2)}' for i in range(14)]

    # Extended feature columns including DNN outputs
    feature_columns2 = feature_columns + dnn_cols

    # Load global max-min values for normalization
    mx_mn = pd.read_csv(str(PATHS['scalar_folder'])+"/max_min.csv", index_col=0)
    mx = pd.DataFrame(mx_mn["mx"]).T
    mn = pd.DataFrame(mx_mn["mn"]).T

    # Manually adjust normalization ranges for DNN outputs and AOD
    mx[dnn_cols] = 1000.
    mn[dnn_cols] = 0.
    mx['aod_avg'] = 4.920678139
    mn['aod_avg'] = 0

    # Run the full pipeline for each GOES file with a progress bar
    for file in tqdm(file_list, desc="Processing GOES Files"):
        run_pipeline(file, collocated, hrrr_vars, var_index, feature_columns, feature_columns2, mx, mn, dnn_cols, plot=plot)

    # End timer and report total time
    total_end_time = time.time()
    elapsed_time = total_end_time - total_start_time
    print(f"\nTotal pipeline execution time: {elapsed_time/60:.2f} minutes")
