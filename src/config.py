#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:50:53 2025

@author: asayeed
"""

"""
Configuration file for GOES PM2.5 estimation pipeline.
Defines base paths and constant parameters used across modules.
"""

import os
from pathlib import Path

# Set the base path of the project (one level up from this file)
BASE_PATH = Path(__file__).resolve().parent.parent

# Define important folders used throughout the project
PATHS = {
    "model_folder": BASE_PATH / "Models/v6_bbox2_relu_final_",  # Path to regional DNN models
    "hrrr_folder": BASE_PATH / "HRRR",                           # Path to HRRR meteorological data
    "scalar_folder": BASE_PATH / "Scalars",                     # Path to precomputed scalars and feature metadata
    "output_nc": BASE_PATH / "OUT_NC",                           # Path to save NetCDF output files
    "plots": BASE_PATH / "PLOTS",                                # Path to save output plots
    "goes_folder": BASE_PATH / "GOES"                             # Path to store downloaded GOES data
}

# Define constant values related to model tiling and grid divisions
CONSTANTS = {
    "istep": 354,  # Step size in i-direction for tiling the GOES data
    "jstep": 360,  # Step size in j-direction for tiling the GOES data

    # Indices for the southwest corner of each model region
    "idxI": [0, 0, 0, 0, 354, 354, 354, 354, 354, 708, 708, 708, 708, 708],
    "idxJ": [360, 720, 1080, 1440, 0, 360, 720, 1080, 1440, 0, 360, 720, 1080, 1440]
}