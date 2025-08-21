import numpy as np
from scipy.interpolate import splrep, BSpline
from sklearn.metrics import r2_score

def determine_step_indices(cellDict, cycle):
    '''
    Determine the step indices for a given cycle in the cell dictionary. As EIS data is not available, the 
    step indices are directly determined based on the cycle data. Returns a list of 4 Step_index: 
    [start_rest, charge, charge_rest, discharge] with standard values of [0, 1, 2, 3].
    '''
    stepList = [0, 1, 2, 3]

    return stepList

def get_Ch_Dis_indices(cycleDict = dict(), stepList = list(), getCC = True, includePre = False):
    """
    Extracts the indices for charge and discharge phases from the cycle dictionary.

    Parameters:
    - cycleDict (dict): Dictionary containing cycle data, including 'Step_index' and 'I' (current).
    - stepList (list): List of step indices, where stepList[1] corresponds to charge and stepList[-1] to discharge.
    - getCC (bool): If True, filters the indices to include only constant current (CC) phases.
    - includePre (bool): If True, includes the index immediately preceding the charge/discharge phase.

    Returns:
    - tuple: A tuple containing two arrays: idxCh, idxDis: Indices for the charge and discharge segments.
    """
    idxCh = np.where(cycleDict['Step_index'] == stepList[1])[0]
    idxDis = np.where(cycleDict['Step_index'] == stepList[-1])[0]

    if(getCC):
        chCC_mask = np.where(cycleDict['I'][idxCh] > cycleDict['I'][idxCh][10] - 0.075)[0]
        disCC_mask = np.where(cycleDict['I'][idxDis] < cycleDict['I'][idxDis][10] + 0.075)[0]
        idxCh = idxCh[chCC_mask]
        idxDis = idxDis[disCC_mask]

    if(includePre):
        idxCh = np.insert(idxCh, 0, idxCh[0]-1)
        idxDis = np.insert(idxDis, 0, idxDis[0]-1)

    return(idxCh, idxDis)

def get_unique_keyname_tuple(featDict, keyNo = 3, continueList = ['cycle_life']):
    '''
    Date: 30/4/2024
    This function is to be applied to the processed feature dictionary for returning a list containing
    the unique list of each keyname. Default feature dictionary's keyname: featDict[(mode, feature, cycleNo)]
    where mode is charge/discharge.

    returns keynameUniqueList for easier looping purposes
    '''
    keynameUniqueList = [[] for _ in range(keyNo)]

    for keynameTuple in list(featDict.keys()):
        if keynameTuple in continueList: continue
        for i in range(keyNo):
            if keynameTuple[i] not in keynameUniqueList[i]:
                keynameUniqueList[i].append(keynameTuple[i])

    return keynameUniqueList