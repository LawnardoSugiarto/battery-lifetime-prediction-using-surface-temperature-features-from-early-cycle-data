import numpy as np
import function_utils

"""
This module provides utility functions for processing and analyzing battery cycle data. 
It includes functions for data smoothing, interpolation, anomaly clipping, and handling missing values. 
These utilities are designed to preprocess raw battery data for further analysis and modeling.

Key Functions:
1. apply_cma_interp: Applies centered moving average (CMA) smoothing and interpolation to cycle data.
2. interpolate_data: Performs linear interpolation on given data within specified limits.
3. moving_average_smoothing: Computes a simple moving average with forward or backward mode.
4. centered_moving_average: Calculates a centered moving average with optional padding for missing values.
5. clip_anomaly: Clips anomalies in data based on a specified threshold.
6. fill_nan_with_left: Fills NaN values in an array by propagating the last valid value to the left.
7. fill_nan_with_right: Fills NaN values in an array by propagating the next valid value to the right.

These functions are primarily used in battery data preprocessing pipelines, such as smoothing temperature 
or voltage profiles, interpolating missing data, and handling anomalies in time-series data.
"""

def apply_cma_interp(cycleDict = dict(), idxArr = np.ndarray, window_size_lst = [11, 21], k_lst = [3, 6],
                   lengthThreshold = 222, xLimitList = [3.2, 4.2], stepInstance = 0.01,
                   xpKey = 'V', fpKey = 'T_cell'):
    '''
        Inputs cycle data battDict[battCode]['cycles'][cycleNo], index array, lists of window_size and k, threshold,
        and xp, fp, xLimitList, stepInstance for interpolation
    '''
    if len(idxArr) < lengthThreshold: k, window_size = (k_lst[0], window_size_lst[0])
    else: k, window_size = (k_lst[1], window_size_lst[1])

    cma = centered_moving_average(data = cycleDict[fpKey][idxArr], window_size = window_size, fillna = True, k = k)
    xp = cycleDict[xpKey][idxArr]
    xLinear, cma_interp = interpolate_data(xp = xp, fp = cma, xLimitList=xLimitList, stepInstance=stepInstance)
    
    return xLinear, cma_interp

def interpolate_data(xp = np.ndarray, fp = np.ndarray, xLimitList = [3.2, 4.2], stepInstance = 0.01):
    '''
    '''
    xLinear = np.arange(xLimitList[0], xLimitList[1], np.sign(np.diff(xLimitList))[0] * stepInstance)

    if function_utils.is_sorted(xLinear): f = np.interp(x = xLinear, xp = xp, fp = fp)
    else: f = np.interp(x = xLinear[::-1], xp = xp[::-1], fp = fp[::-1])[::-1]
        
    return (xLinear, f)

def moving_average_smoothing(X, k, left=True):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if(left):
            if t < k: S[t] = np.mean(X[:t+1])
            else: S[t] = np.sum(X[t-k+1:t+1]) / k
        else:
            if (X.shape[0] - 1 - t) < k: S[t] = np.mean(X[t:])
            else: S[t] = np.sum(X[t:t+k]) / k
    return S       

def centered_moving_average(data, window_size, fillna = False, k = 3):
    """
    Calculate the centered moving average of a given data series.
    
    Parameters:
    data (list or np.array): The input data series.
    window_size (int): The size of the moving window. Must be an odd number.
    
    Returns:
    np.array: The centered moving average series.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    
    half_window = window_size // 2
    cma = np.convolve(data, np.ones(window_size), 'valid') / window_size
    
    # Pad the result to match the original data length
    if(fillna):
        pad_left = moving_average_smoothing(X = data[:half_window], k = k, left = True)
        pad_right = moving_average_smoothing(X = data[-half_window:], k = k, left = False)
    else:
        pad_left = [np.nan] * half_window
        pad_right = [np.nan] * half_window
        
    cma_padded = np.concatenate((pad_left, cma, pad_right))
    
    return cma_padded

def clip_anomaly(arr, threshold = 1):
    # Split indexes
    idxAllArr = np.argwhere(np.abs(np.diff(arr))>threshold)
    idxStartArr = idxAllArr[np.where(idxAllArr <= 40)]
    idxEndArr = idxAllArr[np.where(idxAllArr > 40)]

    for idxEnd in idxEndArr: arr[idxEnd+1] = arr[idxEnd]
    for idxStart in idxStartArr[::-1]: arr[idxStart] = arr[idxStart+1]

    return arr

def fill_nan_with_left(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    arr = arr[idx]
    return arr

def fill_nan_with_right(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), mask.size - 1)
    idx = np.minimum.accumulate(idx[::-1], out = idx[::-1])
    arr = arr[idx[::-1]]
    return arr