import numpy as np
"""
This module provides utility functions for general-purpose operations.
These functions are designed to support various data processing tasks in battery analysis or other applications.

Key Functions:
1. find_nearest: Finds the indices of the array elements closest to specified values, with options for absolute or directional matching.
2. is_sorted: Checks if a given list or array is sorted in ascending order.
3. sort_target_with_ref: Sorts a target array based on the sorted order of a reference array (e.g., voltage).

These functions are versatile and can be used in preprocessing, data alignment, and other computational tasks.
"""

def find_nearest(array, values, abs=True, increasing=True):
    '''
    This function finds the index of the array CLOSEST to selected value/-s
    arg array should be array type, while values should be list type
    Have two modes, decided by argument 'abs'
    If abs==true, we use the minimum absolute value between array and value to return
    else, we simply use the first/last index value using inequality of the value for array
    '''
    array = np.asarray(array)
    indexList = []
    for value in values:
        if (abs==True):
            indexList.append((np.abs(array-value)).argmin())
        else:
            if (increasing==True):
                idx = np.where(array > value)[0][0]
            else:
                idx = np.where(array < value)[0][0]
            indexList.append(idx)

    return np.array(indexList)

def is_sorted(lst):
    if lst is not list: lst = list(lst)
    return lst == sorted(lst)

def sort_target_with_ref(ref, target):
    return (np.array(sorted(ref)), np.array([i for _, i in sorted(zip(ref, target))]))
