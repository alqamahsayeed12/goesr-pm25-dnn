#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:12:50 2025

@author: asayeed
"""


"""
Utility functions used for normalization, custom loss, and spatial smoothing.
"""

# src/utils.py
import pandas as pd
import numpy as np
import keras as k
import keras.losses

def normalize(DF,mx,mn):    
    """
    Normalize input dataframe based on provided max and min values.
    """
    range1 = (mx.values-mn.values)
    d = pd.DataFrame(((DF.values - mn.values)/range1),columns=DF.columns,index=DF.index)
    return d


###--- 2.3 Customized Loss Function used to train model ---###
def custom_loss1(o,p):
    
    """
    Customized IOA-based loss function for Keras models.
    """
    ioa = 1 -(k.sum((o-p)**2))/(k.sum((k.abs(p-k.mean(o))+k.abs(o-k.mean(o)))**2))
    return (-ioa)

### Define Loss
# keras.losses.customLoss1 = customLoss1

###--- 2.4. Function to get spatial n x n (3 x 3 ; in this case ) average ---###
def moving_average_with_nan_2D(arr,w=3):
    
    """
    Calculate a spatial moving average (w x w window) for 2D arrays while properly handling NaNs.
    """
    
    I,J = np.shape(arr)
    
    # Step-1 Processing along columns
    
    # add padding
    pad = w-1
    arr2 = np.zeros((I+pad,J+pad))*np.nan
    arr2[1:-1,1:-1]=arr
    
    #Create mask
    
    mx = np.ma.masked_array(arr2,np.isnan(arr2))
    
    #Sum along axis-1
    ret = np.cumsum(mx.filled(0) ,axis=1)
    
    #count and sum along axis
    counts = np.cumsum(~mx.mask, axis=1)
    
    #create empty variables
    sum_    = np.zeros((I,J))*np.nan
    counts2 = np.zeros((I,J))*np.nan
    
    for j in range(J):
        sum_[:,j] = (ret[:,j+2]-ret[:,max(0,j-1)])[1:-1]
        counts2[:,j] = (counts[:,j+2]-counts[:,max(0,j-1)])[1:-1]
    
    #Step 2 Processing along rows
    arr2 = np.zeros((I+pad,J+pad))*np.nan
    arr2[1:-1,1:-1]=sum_
    mx = np.ma.masked_array(arr2,np.isnan(arr2))
    
    k2 = np.zeros((I+2,J+2))*np.nan
    k2[1:-1,1:-1]=counts2
    mx2 = np.ma.masked_array(k2,np.isnan(k2))
    
    ret = np.cumsum(mx.filled(0) ,axis=0)
    counts = np.cumsum(mx2.filled(0) ,axis=0)
    
    for i in range(I):
        sum_[i,:] = (ret[i+2,:]-ret[max(0,i-1),:])[1:-1]
        counts2[i,:] = (counts[i+2,:]-counts[max(0,i-1),:])[1:-1]
    
    avg = sum_/counts2
    return avg