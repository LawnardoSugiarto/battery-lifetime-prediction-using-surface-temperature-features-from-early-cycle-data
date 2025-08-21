import function_utils

import numpy as np

def create_bat_index_dict(bat_dict_keys, bat_dict):
    """
    This function processes battery cycling data to extract indices for charge, discharge, and rest phases.
    It performs sanity checks to identify potential issues in the data and handles anomalies manually
    for specific cells and cycles. The extracted indices are stored in a nested dictionary for further analysis.

    Inputs:
    - bat_dict_keys: List of battery cell identifiers (bat_dict.keys()).
    - bat_dict: Dictionary containing battery cycling data, including current, capacity, and cycle information.

    Output:
    - bat_index_dict: Nested dictionary containing indices for charge, discharge, and rest phases for each cell and cycle.
    """
    # Initialize the nested dictionary to store indices
    bat_index_dict = function_utils.Vividict()

    # Iterate over each battery cell
    for cellNo in bat_dict_keys:
        for cycleNo in bat_dict[cellNo]['cycles'].keys(): # Iterate over each cycle for the current cell
            # Skip cycle '0' as it is not relevant for processing
            if(cycleNo=='0'): 
                continue

            # Extract data for the current cycle
            checkCell = bat_dict[cellNo]['cycles'][cycleNo]

            # Identify discharge indices where Qd > 0 and current (I) < 0
            idxDis = np.where((checkCell['Qd'] > 0.000001) & (checkCell['I'] < -0.001))[0]
            # Exclude early data points (indices <= 20)
            idxDis = idxDis[idxDis>20]

            # Identify charge indices as all points before the first discharge index
            idxCh = np.arange(0, idxDis[0], 1)

            # Identify rest indices where current is zero and capacity is within a specific range
            idxRest = np.where((checkCell['I'][idxCh] == 0) & 
                               (checkCell['Qc'][idxCh] < 1.015) & 
                               (checkCell['Qc'][idxCh] > 0.4))[0]
            
            # Handle discontinuities in rest indices
            if(np.any(np.diff(idxRest) > 5)):
                if(len(idxRest[0:np.argmax(np.diff(idxRest))+1]) > len(idxRest[np.argmax(np.diff(idxRest))+1:])):
                    idxRest = idxRest[0:np.argmax(np.diff(idxRest))+1]
                else:
                    idxRest = idxRest[np.argmax(np.diff(idxRest))+1:]

            # Perform sanity checks on the data
            if np.mean(checkCell['Qd'][idxCh]) > 0.1: 
                print("The mean Qd for charging index is higher than 0.1 for", cellNo, cycleNo)
            if checkCell['Qd'][idxDis[0]] > 0.03: 
                print("The starting Qd is higher than 0.03 for", cellNo, cycleNo)
            if np.max(checkCell['I'][idxDis]) >= 0: 
                print("There is a positive current in the discharge index", cellNo, cycleNo)
            if(checkCell['I'][idxDis[0]+1]) > 0: 
                print("Wrong starting index for discharge", cellNo, cycleNo)
            if np.any(checkCell['I'][idxCh[0:50]] < -0.5): 
                print("There's a discharge current in the charge index for ", cellNo, cycleNo)
            if np.any(checkCell['I'][idxDis] > 0): 
                print("There's a charge current in the discharge index for ", cellNo, cycleNo)
            if (len(np.where(checkCell['I'][idxCh] == 0)[0]) > 5) & (len(idxRest) < 3): 
                print("Length of idxRest array is < 3 for", cellNo, cycleNo)
            if (len(idxRest)==0): 
                print("Rest index is empty for", cellNo, cycleNo)
            if np.any(np.diff(checkCell['Qc'][idxRest]) > 0.00001): 
                print("Check your rest index as the difference of Qc is > 0 for", cellNo, cycleNo)

            # Append to return dict
            bat_index_dict[cellNo][cycleNo]['rest'] = idxRest
            bat_index_dict[cellNo][cycleNo]['discharge'] = idxDis
            bat_index_dict[cellNo][cycleNo]['charge'] = idxCh

    

    # for manual fix
    # Some cells like b2c12 (cycle 252), b2c27 (cycle 118) and b2c44 (cycle 247) have a weird jump of Qd ~1Ah or more out of nowhere
    ##anomaly; no rest in b1c45 cycle 270. This maybe caused by the batch resumption.
    bat_index_dict['b1c45']['270']['rest'] = bat_index_dict['b1c45']['270']['charge'][143:148]

    ## Anomaly for b2c44 cycle 247 case
    bat_index_dict['b2c44']['247']['rest'] = np.arange(97, 183, 1)
    bat_index_dict['b2c44']['247']['discharge'] = bat_index_dict['b2c44']['247']['discharge'][2:]
    bat_index_dict['b2c44']['247']['charge'] = np.arange(0 , bat_index_dict['b2c44']['247']['discharge'][0], 1)

    return bat_index_dict