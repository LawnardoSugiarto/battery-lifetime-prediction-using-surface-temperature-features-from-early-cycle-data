import numpy as np

"""
This module provides utility functions for battery data processing, dictionary keyname extraction,
etc. It includes methods for error calculation, data interpolation, and assigning step indices
to battery cycling data. These functions are used across various modules to preprocess
data and facilitate feature generation.

Functions:
- root_mean_squared_percentage_error: Calculates the RMSPE for model evaluation.
- interpolate_data: Interpolates battery data based on voltage or other reference values.
- set_stepIndex: Assigns step indices to battery cycling data for easier partitioning.
"""

def root_mean_squared_percentage_error(y_true, y_pred):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    
    return rmspe

def interpolate_data(xLimitList = [3.6, 2.04], step = 500, cellDictOri = dict(), yKeyList = ['Qd', 'T_cell'], xKey = 'V', idxList = list):
    """
    Interpolates battery data based on a reference key (`xKey`) and generates linearized arrays for specified keys (`yKeyList`). 

    Inputs:
    - xLimitList: List specifying the start and end limits for interpolation (default: [3.6, 2.04]).
    - step: Number of interpolation steps (default: 500).
    - cellDictOri: Original dictionary containing battery data.
    - yKeyList: List of keys to interpolate (default: ['Qd', 'T_cell']).
    - xKey: Reference key for interpolation (default: 'V').
    - idxList: List of indices to use for interpolation.

    Output:
    - returnDict: Dictionary containing interpolated data for specified keys.
    """
    change = 1
    cellDict = cellDictOri.copy()
    returnDict = {}

    # Determine interpolation direction based on xLimitList
    if xLimitList[0] > xLimitList[1]: 
        linearArr = -np.linspace(xLimitList[0], xLimitList[1], step)
        xArr = -cellDict[xKey][idxList]
        change = -1
    else: 
        linearArr = np.linspace(xLimitList[0], xLimitList[1], step)
        xArr = cellDict[xKey][idxList]
    
    # Interpolate data for each key in yKeyList
    for yKey in yKeyList:
        # Separate temperature data based on current mode (charge/discharge)
        if 'T' in yKey:
            # By checking current 'I'
            if np.mean(cellDict['I'][idxList[0:int(len(idxList)/2)]]) < 0: 
                mode = 'd' # Discharge mode
            else: 
                mode = 'c' # Charge mode
            
            returnDict[yKey + mode + 'lin'] = np.interp(linearArr, xArr, cellDict[yKey][idxList])
        else:
            returnDict[yKey + 'lin'] = np.interp(linearArr, xArr, cellDict[yKey][idxList])

    # Add interpolated reference key to the dictionary
    returnDict[xKey + 'lin'] = change * linearArr
        
    return returnDict

def set_stepIndex(bat_dict, bat_index_dict_ori, stepIndexList = [[1, 8, 9, 10], [1, 3, 4, 5]]):
    '''
    Assigns step indices to the battery cycling data dictionary (`bat_dict`) for easier data partitioning. 
    This function is particularly useful for datasets that do not include step indices originally.

    Inputs:
    - bat_dict: Dictionary containing battery cycling data (should be a Vividict).
    - bat_index_dict_ori: Original dictionary to store step indices.
    - stepIndexList: List of two lists specifying step indices for different modes (default: [[1, 8, 9, 10], [1, 3, 4, 5]]).
      List 1 corresponds to cyc0_woEIS, and List 2 corresponds to woEIS. Each list contains:
      1. Rest (before charge), 2. Charge, 3. Rest (before discharge), and 4. Discharge.

    Output:
    - bat_index_dict_ori: Updated dictionary with assigned step indices.

    NOTE: bat_dict should be a Vividict
    Last modified: 16/5/2023
    '''
    
    bat_index_dict = bat_index_dict_ori.copy()
    
    for datatype in bat_dict.keys():
        for cellno in bat_dict[datatype].keys(): # Iterate over each battery cell in bat_dict
            for cycle in bat_dict[datatype][cellno]['cycles'].keys(): # Iterate over each cycle for the current cell
                # To shorten the pointer
                cycleData = bat_dict[datatype][cellno]['cycles'][cycle]
                indexData = bat_index_dict[cellno][cycle]
                # Skip cycle '0' as it is not relevant for processing
                if cycle == '0': continue
                else: indexList = stepIndexList[1]
                
                lenArr = cycleData['t'].shape[0]
                
                # Assign step indices based on stepIndexList
                newArr = np.full(lenArr, indexList[0])
                newArr[indexData['charge']] = indexList[1]
                newArr[indexData['rest']] = indexList[2]
                newArr[indexData['discharge']] = indexList[3]
                newArr[indexData['discharge'][-1]+1:] = indexList[3] + 1
                cycleData['Step_index'] = newArr
                
                # Additional fix
                cycleData['T_cell'] = cycleData['T']
            
    return bat_dict

# Function to find the index of the array closest to a selected value/-s
def find_nearest(array, values, abs=True, increasing=True):
    '''
    This function finds the index of the array CLOSEST to selected value/-s
    arg array should be array type, while values should be list type
    Have two modes, decided by argument 'abs'
    If abs==true, we use the minimum absolute value between array and value to return
    else, we simply use the first/last index value using inequality of the value for array
    '''
    array = np.asarray(array)
    index_array = []
    for value in values:
        if (abs==True):
            index_array.append((np.abs(array-value)).argmin())
        else:
            if (increasing==True):
                idx = np.where(array > value)[0][0]
            else:
                idx = np.where(array < value)[0][0]
            index_array.append(idx)

    return index_array

def get_unique_keyname_tuple(feat_dict, key_no = 3, continue_list = ['cycle_life']):
    """
    Extracts unique values for each key in a tuple from the keys of a feature dictionary (`feat_dict`).
    The function returns a list of lists, where each sublist contains the unique values for a specific key position
    in the tuple. It skips keys that are in the `continue_list`.
    """
    # Initialize a list of empty lists to store unique values for each key position
    keyname_unique = [[] for _ in range(key_no)]

    # Iterate over all keys in the feature dictionary
    for keyname_tuple in list(feat_dict.keys()):
        # Skip keys that are in the continue_list
        if keyname_tuple in continue_list: 
            continue

        # Iterate over each key position in the tuple
        for i in range(key_no):
            # Add the value to the corresponding sublist if it is not already present
            if keyname_tuple[i] not in keyname_unique[i]:
                keyname_unique[i].append(keyname_tuple[i])
                
    return keyname_unique

class Vividict(dict):
    '''
    This function creates a dictionary that retain local pointer to value
    The return type(self) gives a faster and trackable return compared to dictionary lookup
    '''
    def __missing__(self, key):
        value = self[key] = type(self)() # retain local pointer to value
        return value                     # faster to return than dict lookup
