import function_utils

from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis
import numpy as np

"""
This module provides utility functions for generating statistical features from time-series
battery data. It includes methods for calculating statistical and cycle-based features, as well 
as handling battery data dictionaries. The features are used for analyzing and predicting
battery cycle life.

Key Features:
- `T_Q` and `dTdQ`: Proposed statistical temperature features in our work.
- `nature` and `nature_2to10`: Features derived from Severson et al.'s work (published in Nature Energy),
  using data from the original 100 cycles and the first 10 cycles (i.e., 2 to 10).
"""

def create_T_Q_lin_feature(bat_dict, cycle_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '49', '99', '199'], wCV = False, avgUpToCycle = 10):
    '''
    This function extracts cycling temperature data at selected capacity instances for each cycle in `cycle_list`.
    It evaluates our proposed statistical features such as variance, mean, skewness, kurtosis, maximum, minimum, and range (max-min)
    for both charge and discharge modes. The extracted features are stored in a nested dictionary for easier data slicing.

    Inputs:
    - bat_dict: Dictionary containing battery data
    - cycle_list: List of cycle identifiers (default: ['1', '2', ..., '199'])
    - wCV: Boolean flag indicating whether to include CV (constant voltage) data (default: False)
    - avgUpToCycle: Integer specifying the number of cycles to average features over (default: 10)

    Output:
    - dictionary: A nested dictionary containing calculated temperature features.
        
    Last modified: 26/9/2023
    '''
    # Initialize feature dictionary
    dictionary = function_utils.Vividict()

    # Define feature keys
    keyname_list = ['T_arr', 'Q_arr', 'Variance', 'Mean', 'Skewness', 'Kurtosis', 'Maximum', 'Minimum', 'Max-Min']
    keyname_list_ori = ['Maximum', 'Minimum', 'Max-Min', 'Variance', 'Mean', 'Skewness', 'Kurtosis']
    
    # 4/4/2023: Define cycle groups for older and younger cycles for dT(Q) operation
    cycOld_list = ['9', '49', '99', '149']
    cycYoung_list = ['1','4']

    # Define modes for processing (charge and discharge)
    mode_list = ['Charge', 'Discharge']
    
    # Iterate over modes (charge/discharge)
    for mode in mode_list: 
        for keyname in keyname_list: # Iterate over feature keys
            for cycle in cycle_list: # Iterate over cycles in cycle_list
                # Create nested dictionaries for easier data slicing
                if(keyname in keyname_list_ori): 
                    dictionary[mode, keyname, cycle] = np.array([])
                else: 
                    dictionary[mode, keyname, cycle] = function_utils.Vividict()
                
            # Keyname for averaging features up to the iterated values, avgUpToCycle
            if(keyname in keyname_list_ori):
                for cycleEnd in range(2, avgUpToCycle): dictionary[mode, keyname, 'Average_2to'+str(cycleEnd+1)] = np.array([])
        
                
        # First create the arrays for dT(Q) of the older and younger cycles
        for cycOld in cycOld_list:
            for cycYoung in cycYoung_list:
                for keyname in keyname_list_ori + ['T_arrDiff']:
                    if (keyname in keyname_list_ori): dictionary[mode, keyname, cycOld+'-'+cycYoung] = np.array([])
                    else: dictionary[mode, keyname, cycOld+'-'+cycYoung] = function_utils.Vividict()
                    
    # Target: cycle_life
    dictionary['cycle_life'] = np.array([])

    # Process battery data
    for datatype in bat_dict.keys():
        for cellno in bat_dict[datatype].keys():
            # To simplify and shorten using variables
            cell_dict = bat_dict[datatype][cellno]
            cell_cycles = cell_dict['cycles']

            # Iterate for each mode (i.e., charge/discharge)
            for i, mode in enumerate(mode_list):
                for cycle in cycle_list: # Over all cycles
                    # Shorten the address
                    cell = cell_cycles[cycle]

                    # To determine the keyMode for charge or discharge using i, enumerate
                    if(2*i+1 == 1): 
                        cycMode = 'clin'
                        xRef = 'Qclin'
                    elif(2*i+1 == 3): 
                        cycMode = 'dlin'
                        if wCV == False: xRef = 'Vlin'
                        else: xRef = 'Qdlin'
                        
                    # Get the raw time-series data
                    # We modified in 4/4/2023 to use T_cellclin and T_celldlin, processed beforehand in modelTraining_xx.ipynb
                    T_arr = cell['T_cell' + cycMode]
                    Q_arr = cell[xRef]
                    dictionary[mode, 'T_arr', cycle][datatype + '_' + cellno] = T_arr
                    dictionary[mode, 'Q_arr', cycle][datatype + '_' + cellno] = Q_arr

                    # Evaluate statistical features and append for each cycle
                    dictionary[mode, 'Variance', cycle] = np.append(dictionary[mode, 'Variance', cycle], np.log10(np.abs(np.var(T_arr))))
                    dictionary[mode, 'Mean', cycle] = np.append(dictionary[mode, 'Mean', cycle], np.log10(np.abs(np.mean(T_arr))))
                    dictionary[mode, 'Skewness', cycle] = np.append(dictionary[mode, 'Skewness', cycle], np.log10(np.abs(skew(T_arr))))
                    dictionary[mode, 'Kurtosis', cycle] = np.append(dictionary[mode, 'Kurtosis', cycle], np.log10(np.abs(kurtosis(T_arr))))
                    dictionary[mode, 'Maximum', cycle] = np.append(dictionary[mode, 'Maximum', cycle], np.log10(np.abs(np.max(T_arr))))
                    dictionary[mode, 'Minimum', cycle] = np.append(dictionary[mode, 'Minimum', cycle], np.log10(np.abs(np.min(T_arr))))
                    
                    # For charge, we aim to take minimal temperature value around the first 50 data points, while for discharge, we take the whole array
                    if(mode == 'Charge'):
                        dictionary[mode, 'Max-Min', cycle] = np.append(dictionary[mode, 'Max-Min', cycle], np.log10(np.abs(np.max(T_arr) - np.min(T_arr[:50]))))
                    else:
                        dictionary[mode, 'Max-Min', cycle] = np.append(dictionary[mode, 'Max-Min', cycle], np.log10(np.abs(np.max(T_arr) - np.min(T_arr))))
                    
                # Create for two cycles difference; 5/1/2023
                for cycOld in cycOld_list:
                    for cycYoung in cycYoung_list:
                        # Calculate difference using numpy array operation
                        T_arrDiff = np.array(np.array(dictionary[mode, 'T_arr', cycOld][datatype + '_' + cellno]) - np.array(dictionary[mode, 'T_arr', cycYoung][datatype + '_' + cellno]))
                        
                        # Assign each statistical features as well as T_listDiff for checking purposes later
                        # Note: this was not used in our final publication, but we kept it for future reference
                        dictionary[mode, 'T_arrDiff', cycOld+'-'+cycYoung][datatype + '_' + cellno] = T_arrDiff
                        dictionary[mode, 'Variance', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Variance', cycOld+'-'+cycYoung], np.log10(np.abs(np.var(T_arrDiff))))
                        dictionary[mode, 'Mean', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Mean', cycOld+'-'+cycYoung], np.log10(np.abs(np.mean(T_arrDiff))))
                        dictionary[mode, 'Skewness', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Skewness', cycOld+'-'+cycYoung], np.log10(np.abs(skew(T_arrDiff))))
                        dictionary[mode, 'Kurtosis', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Kurtosis', cycOld+'-'+cycYoung], np.log10(np.abs(kurtosis(T_arrDiff))))
                        dictionary[mode, 'Maximum', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Maximum', cycOld+'-'+cycYoung], np.log10(np.abs(np.max(np.abs(T_arrDiff)))))
                        dictionary[mode, 'Minimum', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Minimum', cycOld+'-'+cycYoung], np.log10(np.abs(np.min(np.abs(T_arrDiff)))))
                        dictionary[mode, 'Max-Min', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Max-Min', cycOld+'-'+cycYoung], np.log10(np.abs(np.max(np.abs(T_arrDiff)) - np.min(np.abs(T_arrDiff)))))
                    
            # Append each cell's cycle life
            dictionary['cycle_life'] = np.append(dictionary['cycle_life'], cell_dict['cycle_life'])
        
    # Make average of the above features (4/4/2023)
    for mode in mode_list:
        for keyname_avg in keyname_list_ori:
            for cycleEnd in range(2, avgUpToCycle):
                sum = 0
                for cycle_avg in cycle_list[:cycleEnd]: sum += dictionary[mode, keyname_avg, cycle_avg]
                keyNameAvg = 'Average_2to' + str(cycleEnd+1)
                dictionary[mode, keyname_avg, keyNameAvg] = np.append(dictionary[mode, keyname_avg, keyNameAvg], sum/cycleEnd)

    return dictionary

def create_dTdQ_lin_feature(bat_dict, cycle_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '49', '99', '199'], wCV = False, avgUpToCycle = 10):
    """
    This function extracts cycling temperature data at selected capacity instances for each cycle in `cycle_list`.
    It calculates the derivative of temperature with respect to capacity (`dT/dQ`) and evaluates statistical features
    such as variance, mean, skewness, kurtosis, maximum, minimum, and range (max-min) for both charge and discharge modes.
    The extracted features are stored in a nested dictionary for easier data slicing.

    Inputs:
    - bat_dict: Dictionary containing battery data.
    - cycle_list: List of cycle identifiers (default: ['1', '2', ..., '199']).
    - wCV: Boolean flag indicating whether to include CV (constant voltage) data (default: False).
    - avgUpToCycle: Integer specifying the number of cycles to average features over (default: 10).

    Output:
    - dictionary: A nested dictionary containing calculated `dT/dQ` features.
   
    Last modified: 26/9/2023
    """
    # Initialize feature dictionary
    dictionary = function_utils.Vividict()

    # Define feature keys
    keyname_list = ['T_arr', 'Q_arr', 'dTdQ_arr', 'Variance', 'Mean', 'Skewness', 'Kurtosis', 'Maximum', 'Minimum', 'Max-Min']
    keyname_list_ori = ['Maximum', 'Minimum', 'Max-Min', 'Variance', 'Mean', 'Skewness', 'Kurtosis']
    
    # 4/4/2023: Define cycle groups for older and younger cycles for d(dT/dQ)(Q) operations
    cycOld_list = cycle_list[9:]
    cycYoung_list = ['1', '4']

    # Define modes for processing (charge and discharge)
    mode_list = ['Charge', 'Discharge']
    
    # Iterate over modes (charge/discharge)
    for mode in mode_list: 
        for keyname in keyname_list: # Iterate over feature keys
            for cycle in cycle_list: # Iterate over cycles in cycle_list
               # Create nested dictionaries for easier data slicing
                if(keyname in keyname_list_ori): dictionary[mode, keyname, cycle] = np.array([])
                else: dictionary[mode, keyname, cycle] = function_utils.Vividict()

            # Keyname for averaging features up to the iterated values, avgUpToCycle
            if(keyname in keyname_list_ori):
                for cycleEnd in range(2, avgUpToCycle): dictionary[mode, keyname, 'Average_2to'+str(cycleEnd+1)] = np.array([])
                
        # First create the arrays for d(dT/dQ)(Q) of the older and younger cycles
        for cycOld in cycOld_list:
            for cycYoung in cycYoung_list:
                for keyname in keyname_list_ori + ['dTdQ_arrDiff']:
                        if (keyname in keyname_list_ori): dictionary[mode, keyname, cycOld+'-'+cycYoung] = np.array([])
                        else: dictionary[mode, keyname, cycOld+'-'+cycYoung] = function_utils.Vividict()
    
    # Target: cycle_life
    dictionary['cycle_life'] = np.array([])

    # Process battery data
    for datatype in bat_dict.keys():
        for cellno in bat_dict[datatype].keys():
            # To simplify and shorten using variables
            cell_dict = bat_dict[datatype][cellno]
            cell_cycles = cell_dict['cycles']

            # Iterate for each mode (i.e., charge/discharge)
            for i, mode in enumerate(mode_list):
                for cycle in cycle_list: # Over all cycles
                    dTdQ_arr = np.array([])
                    # Shorten the address
                    cell = cell_cycles[cycle]

                    # To determine the keyMode for charge or discharge using i, enumerate
                    if(2*i+1 == 1): 
                        cycMode = 'clin'
                        xRef = 'Qclin'
                    elif(2*i+1 == 3): 
                        cycMode = 'dlin'
                        if wCV == False: xRef = 'Vlin'
                        else: xRef = 'Qdlin'

                    # Get the raw time-series data
                    # We modified in 4/4/2023 to use T_cellclin and T_celldlin, processed beforehand in modelTraining_xx.ipynb
                    T_arr = cell['T_cell' + cycMode]
                    Q_arr = cell[xRef]
                    
                    # Calculate the difference, and put it inside dTdQ_arr
                    for j in range(len(Q_arr)-1):
                        dTdQ = (T_arr[j+1] - T_arr[j]) / (Q_arr[j+1] - Q_arr[j])
                        dTdQ_arr = np.append(dTdQ_arr, dTdQ)
                        
                    # Check for NaN values in dTdQ_arr
                    if True in np.isnan(dTdQ_arr):
                        # Get the list of index(es) of nan values
                        nanIndexArr = np.argwhere(np.isnan(dTdQ_arr)).reshape(1,-1)[0]
                        for nanIndex in np.sort(nanIndexArr):
                            dTdQ_arr[nanIndex] = dTdQ_arr[nanIndex-1] # replace with preceeding element

                    dictionary[mode, 'T_arr', cycle][datatype + '_' + cellno] = T_arr
                    dictionary[mode, 'Q_arr', cycle][datatype + '_' + cellno] = Q_arr
                    dictionary[mode, 'dTdQ_arr', cycle][datatype + '_' + cellno] = dTdQ_arr
                     # Evaluate statistical features and append for each cycle
                    dictionary[mode, 'Variance', cycle] = np.append(dictionary[mode, 'Variance', cycle], np.log10(np.abs(np.var(dTdQ_arr))))
                    dictionary[mode, 'Mean', cycle] = np.append(dictionary[mode, 'Mean', cycle], np.log10(np.abs(np.mean(dTdQ_arr))))
                    dictionary[mode, 'Skewness', cycle] = np.append(dictionary[mode, 'Skewness', cycle], np.log10(np.abs(skew(dTdQ_arr))))
                    dictionary[mode, 'Kurtosis', cycle] = np.append(dictionary[mode, 'Kurtosis', cycle], np.log10(np.abs(kurtosis(dTdQ_arr))))
                    dictionary[mode, 'Maximum', cycle] = np.append(dictionary[mode, 'Maximum', cycle], np.log10(np.abs(np.max(dTdQ_arr))))
                    dictionary[mode, 'Minimum', cycle] = np.append(dictionary[mode, 'Minimum', cycle], np.log10(np.abs(np.min(dTdQ_arr))))
                    dictionary[mode, 'Max-Min', cycle] = np.append(dictionary[mode, 'Max-Min', cycle], np.log10(np.abs(np.max(dTdQ_arr) - np.min(dTdQ_arr))))
                    
                # Create for two cycles difference; 5/1/2023
                for cycOld in cycOld_list:
                    for cycYoung in cycYoung_list:
                        # Calculate difference using numpy array operation           
                        dTdQ_arrDiff = np.array(np.array(dictionary[mode, 'dTdQ_arr', cycOld][datatype + '_' + cellno]) - np.array(dictionary[mode, 'dTdQ_arr', cycYoung][datatype + '_' + cellno]))
                        
                        # Assign each statistical features as well as dTdQ_arrDiff for checking purposes later
                        # Note: this was not used in our final publication, but we kept it for future reference
                        dictionary[mode, 'dTdQ_arrDiff', cycOld+'-'+cycYoung][datatype + '_' + cellno] = dTdQ_arrDiff
                        dictionary[mode, 'Variance', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Variance', cycOld+'-'+cycYoung], np.log10(np.abs(np.var(dTdQ_arrDiff))))
                        dictionary[mode, 'Mean', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Mean', cycOld+'-'+cycYoung], np.log10(np.abs(np.mean(dTdQ_arrDiff))))
                        dictionary[mode, 'Skewness', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Skewness', cycOld+'-'+cycYoung], np.log10(np.abs(skew(dTdQ_arrDiff))))
                        dictionary[mode, 'Kurtosis', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Kurtosis', cycOld+'-'+cycYoung], np.log10(np.abs(kurtosis(dTdQ_arrDiff))))
                        dictionary[mode, 'Maximum', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Maximum', cycOld+'-'+cycYoung], np.log10(np.abs(np.max(np.abs(dTdQ_arrDiff)))))
                        dictionary[mode, 'Minimum', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Minimum', cycOld+'-'+cycYoung], np.log10(np.abs(np.min(np.abs(dTdQ_arrDiff)))))
                        dictionary[mode, 'Max-Min', cycOld+'-'+cycYoung] = np.append(dictionary[mode, 'Max-Min', cycOld+'-'+cycYoung], np.log10(np.abs(np.max(np.abs(dTdQ_arrDiff)) - np.min(np.abs(dTdQ_arrDiff)))))
                    
            # Append each cell's cycle life
            dictionary['cycle_life'] = np.append(dictionary['cycle_life'], cell_dict['cycle_life'])

    # Make average of the above features (4/4/2023)
    for mode in mode_list:
        for keyname_avg in keyname_list_ori:
            for cycleEnd in range(2, avgUpToCycle):
                sum = 0
                for cycle_avg in cycle_list[:cycleEnd]: sum += dictionary[mode, keyname_avg, cycle_avg]
                keyNameAvg = 'Average_2to' + str(cycleEnd+1)
                dictionary[mode, keyname_avg, keyNameAvg] = np.append(dictionary[mode, keyname_avg, keyNameAvg], sum/cycleEnd)

    return dictionary

# New version:17/5/2023
def create_nature_feature_v2(bat_index_dict, bat_dict, cycle_list=['99','9'], V_cutoff_list = [3.6, 1.99]):
    """
    The function processes battery data dictionaries and returns a feature dictionary.
    The features were used in Severson et al. work in Nature Energy, for building their
    three models: Variance, Discharge and Full models. Each model contain different subset of
    the features generated here, extracted from the first 100 cycling data.
    
    Inputs:
    - bat_index_dict: Dictionary mapping battery cycling steps (e.g. charge, rest, discharge)
    - bat_dict: Dictionary containing battery data
    - cycle_list: List of cycle identifiers (default: ['99', '9']) used for dQ(V)
    - V_cutoff_list: Voltage cutoff range (default: [3.6, 1.99]) for CC-segment
    Output:
    - dictionary: A nested dictionary containing calculated features.
    """
    # Initialize feature dictionary
    dictionary = function_utils.Vividict()
    V_list = np.arange(V_cutoff_list[0], V_cutoff_list[1],-0.02)

    # Define feature keys
    keyname_list = ['V_100_arr', 'Q_100_arr', 'V_10_arr', 'Q_10_arr','Q_V_arrDiff',   
                    'Q_V_Variance', 'Q_V_Mean', 'Q_V_Skewness', 'Q_V_Kurtosis', 'Q_V_Minimum', 'Q_V_2V',
                    'Qd_c2', 'Qd_c100', 'Qd_max-c2', 'Slope_2to100', 'Intercept_2to100', 'Slope_91to100', 'Intercept_91to100',
                    'Avg_charge_time', 'IR_cycle2', 'Min_IR', 'IR_cycle100-2', 'MaxT_2to100', 'MinT_2to100', 'IntegralT_2to100',
                    'cycle_life']
    
    # Initialize dictionary keys for arrays and nested dictionaries
    for keyname in keyname_list[0:5]: 
        dictionary[keyname] = function_utils.Vividict()
    for keyname in keyname_list[5:]: 
        dictionary[keyname] = np.array([])
    
    # Define anomaly list (used for filtering or special handling)
    anomaly_list = ['b1c0', 'b1c18', 'b2c12', 'b2c44']
    
    # Process battery data
    for datatype in bat_dict.keys():
        for batno in bat_dict[datatype].keys():
            cell_cycles = bat_dict[datatype][batno]['cycles'] 
            cell_summary = bat_dict[datatype][batno]['summary']

            # Calculate Q_V_arrDiff (difference between cycle 100(99) and cycle 10(9) discharge data)
            Q_V_arrDiff = cell_cycles['99']['Qdlin'] - cell_cycles['9']['Qdlin']
            dictionary['Q_V_arrDiff'][batno] = Q_V_arrDiff
            
            # Statistical features for Q(V) 100-10, used in Variance, Discharge and Full models
            dictionary['Q_V_Variance'] = np.append(dictionary['Q_V_Variance'], np.log10(np.abs(np.var(Q_V_arrDiff))))
            dictionary['Q_V_Mean'] = np.append(dictionary['Q_V_Mean'], np.log10(np.abs(np.mean(Q_V_arrDiff))))
            dictionary['Q_V_Skewness'] = np.append(dictionary['Q_V_Skewness'], np.log10(np.abs(skew(Q_V_arrDiff))))
            dictionary['Q_V_Kurtosis'] = np.append(dictionary['Q_V_Kurtosis'], np.log10(np.abs(kurtosis(Q_V_arrDiff))))
            dictionary['Q_V_Minimum'] = np.append(dictionary['Q_V_Minimum'], np.log10(np.abs(min(Q_V_arrDiff))))
            dictionary['Q_V_2V'] = np.append(dictionary['Q_V_2V'], np.log10(np.abs(Q_V_arrDiff[-1])))

            # Q_discharge summary for cycle 1 to 100
            cycleNo = cell_summary['cycle'][0:100].reshape(-1,1)
            QD = cell_summary['QD'][0:100].reshape(-1,1)

            linreg_2to100 = LinearRegression().fit(cycleNo[1:100], QD[1:100])
            linreg_91to100 = LinearRegression().fit(cycleNo[90:100], QD[90:100])

            # Discharge capacity at cycle 2, 100 and difference between max & cycle2, used in Discharge and Full models
            dictionary['Qd_c2'] = np.append(dictionary['Qd_c2'], cell_summary['QD'][1])
            dictionary['Qd_c100'] = np.append(dictionary['Qd_c100'], cell_summary['QD'][99])
            dictionary['Qd_max-c2'] = np.append(dictionary['Qd_max-c2'], np.max(cell_summary['QD']) - cell_summary['QD'][1])
            dictionary['Slope_2to100'] = np.append(dictionary['Slope_2to100'], linreg_2to100.coef_[0][0])
            dictionary['Intercept_2to100'] = np.append(dictionary['Intercept_2to100'], linreg_2to100.intercept_[0])
            dictionary['Slope_91to100'] = np.append(dictionary['Slope_91to100'], linreg_91to100.coef_[0][0])
            dictionary['Intercept_91to100'] = np.append(dictionary['Intercept_91to100'], linreg_91to100.intercept_[0])

            # Other misc. features, used only in Full model
            dictionary['Avg_charge_time'] = np.append(dictionary['Avg_charge_time'], np.mean(cell_summary['chargetime'][1:6])) #first 5 cycles, 2 to 6
            dictionary['IR_cycle2'] = np.append(dictionary['IR_cycle2'], cell_summary['IR'][1]) # cycle 2
            dictionary['Min_IR'] = np.append(dictionary['Min_IR'], np.min(cell_summary['IR'][ [i + 1 for i in list(np.nonzero(cell_summary['IR'][1:100])) ] ])) # cycle 2 to 100
            dictionary['IR_cycle100-2'] = np.append(dictionary['IR_cycle100-2'], cell_summary['IR'][99] - cell_summary['IR'][1])
            dictionary['MaxT_2to100'] = np.append(dictionary['MaxT_2to100'], np.max(cell_summary['Tmax'][1:100]))
            dictionary['MinT_2to100'] = np.append(dictionary['MinT_2to100'], np.min(cell_summary['Tmin'][1:100]))
            dictionary['IntegralT_2to100'] = np.append(dictionary['IntegralT_2to100'], np.sum([np.trapz(cell_cycles[cyc]['T'], cell_cycles[cyc]['t']) for cyc in map(str, np.arange(1,100,1))]))

            # Append cycle life
            dictionary['cycle_life'] = np.append(dictionary['cycle_life'], int(bat_dict[datatype][batno]['cycle_life']))

            # manual fix: change value of 'b1c18' to 'b1c19'
            if batno == 'b1c19':
                dictionary['IntegralT_2to100'][-2] = dictionary['IntegralT_2to100'][-1]
                
            # Qd max-c2
            if batno in anomaly_list:
                Qd_arr = np.sort(cell_summary['QD'])
                dictionary['Qd_max-c2'][-1] = Qd_arr[-2] - dictionary['Qd_c2'][-1]

    return dictionary


def create_nature_feature_v2_2to10(bat_index_dict, bat_dict, cyc=['9','1'], V_cutoff_list = [3.6, 1.99]):
    """
    The function processes battery data dictionaries and returns a feature dictionary.
    The features were used in Severson et al. work in Nature Energy, for building their
    three models: Variance, Discharge and Full models. Each model contain different subset of
    the features generated here.

    In this function, we restrict the usage of data down to the first 10 cycles, instead of the 
    original 100 cycles; for benchmark purposes.
    
    Inputs:
    - bat_index_dict: Dictionary mapping battery cycling steps (e.g. charge, rest, discharge)
    - bat_dict: Dictionary containing battery data
    - cycle_list: List of cycle identifiers (default: ['9', '1']) used for dQ(V)
    - V_cutoff_list: Voltage cutoff range (default: [3.6, 1.99]) for CC-segment
    Output:
    - dictionary: A nested dictionary containing calculated features.
    """
    # Initialize feature dictionary
    dictionary = function_utils.Vividict()

    # Define the cycle key based on the input cycle list
    cycleKey = str(int(cyc[0])+1)

    # Generate voltage range for CC-segment
    V_list = np.arange(V_cutoff_list[0], V_cutoff_list[1],-0.02)

    # Define feature keys
    keyname_list = ['V_'+cycleKey+'_arr', 'Q_'+cycleKey+'_arr', 'V_2_arr', 'Q_2_arr','Q_V_arrDiff',   
                    'Q_V_Variance', 'Q_V_Mean', 'Q_V_Skewness', 'Q_V_Kurtosis', 'Q_V_Minimum', 'Q_V_2V',
                    'Qd_c2', 'Qd_c'+cycleKey, 'Qd_max-c2', 'Slope_2to10', 'Intercept_2to10', 'Slope_6to10', 'Intercept_6to10',
                    'Avg_charge_time', 'IR_cycle2', 'Min_IR', 'IR_cycle'+cycleKey+'-2', 'MaxT_2to'+cycleKey, 'MinT_2to'+cycleKey, 'IntegralT_2to'+cycleKey,
                    'cycle_life']
    
    # Initialize dictionary keys for arrays and nested dictionaries
    for keyname in keyname_list[0:5]: 
        dictionary[keyname] = function_utils.Vividict()
    for keyname in keyname_list[5:]: 
        dictionary[keyname] = np.array([])
    
    
    # Process battery data
    for datatype in bat_dict.keys():
        for batno in bat_dict[datatype].keys():
            cell_cycles = bat_dict[datatype][batno]['cycles'] 
            cell_summary = bat_dict[datatype][batno]['summary']
            
             # Calculate Q_V_arrDiff (difference between cycle 10 and cycle 2 discharge data)
            disIdx_C10_start = bat_index_dict[batno][cyc[0]]['discharge'][0] #cycle 10
            disIdx_C10_end = bat_index_dict[batno][cyc[0]]['discharge'][-1]
            disIdx_C2_start = bat_index_dict[batno][cyc[1]]['discharge'][0] #cycle 2
            disIdx_C2_end = bat_index_dict[batno][cyc[1]]['discharge'][-1]

            Q_C10_idx = function_utils.find_nearest(cell_cycles[cyc[0]]['V'][disIdx_C10_start:disIdx_C10_end], V_list)
            Q_C10_idx = [idx + disIdx_C10_start for idx in Q_C10_idx]

            Q_C2_idx = function_utils.find_nearest(cell_cycles[cyc[1]]['V'][disIdx_C2_start:disIdx_C2_end], V_list)
            Q_C2_idx = [idx + disIdx_C2_start for idx in Q_C2_idx]

            Q_C10 = cell_cycles[cyc[0]]['Qd'][Q_C10_idx]
            Q_C2 = cell_cycles[cyc[1]]['Qd'][Q_C2_idx]
            
            # For checking purpose
            V_C10 = cell_cycles[cyc[0]]['V'][Q_C10_idx]
            V_C2 = cell_cycles[cyc[1]]['V'][Q_C2_idx]

            Q_V_arrDiff = Q_C10 - Q_C2

            # Verification keynames
            dictionary['V_'+cycleKey+'_arr'][batno] = V_C10
            dictionary['Q_'+cycleKey+'_arr'][batno] = Q_C10
            dictionary['V_2_arr'][batno] = V_C2
            dictionary['Q_2_arr'][batno] = Q_C2
            dictionary['Q_V_arrDiff'][batno] = Q_V_arrDiff
            
            # Statistical features for Q(V) 10-2 used in Variance, Discharge and Full models
            dictionary['Q_V_Variance'] = np.append(dictionary['Q_V_Variance'], np.log10(np.abs(np.var(Q_V_arrDiff))))
            dictionary['Q_V_Mean'] = np.append(dictionary['Q_V_Mean'], np.log10(np.abs(np.mean(Q_V_arrDiff))))
            dictionary['Q_V_Skewness'] = np.append(dictionary['Q_V_Skewness'], np.log10(np.abs(skew(Q_V_arrDiff))))
            dictionary['Q_V_Kurtosis'] = np.append(dictionary['Q_V_Kurtosis'], np.log10(np.abs(kurtosis(Q_V_arrDiff))))
            dictionary['Q_V_Minimum'] = np.append(dictionary['Q_V_Minimum'], np.log10(np.abs(min(Q_V_arrDiff))))
            dictionary['Q_V_2V'] = np.append(dictionary['Q_V_2V'], np.log10(np.abs(Q_V_arrDiff[-1])))

            # Q_discharge summary for cycle 1 to 10
            cycleNo = cell_summary['cycle'][0:10].reshape(-1,1)
            QD = cell_summary['QD'][0:10].reshape(-1,1)

            linreg_2to10 = LinearRegression().fit(cycleNo[1:10], QD[1:10])
            linreg_6to10 = LinearRegression().fit(cycleNo[5:10], QD[5:10])

            # Discharge capacity at cycle 2, 10 and difference between max & cycle2 used in Discharge and Full models
            dictionary['Qd_c2'] = np.append(dictionary['Qd_c2'], cell_summary['QD'][int(cyc[1])])
            dictionary['Qd_c'+cycleKey] = np.append(dictionary['Qd_c'+cycleKey], cell_summary['QD'][int(cyc[0])])
            dictionary['Qd_max-c2'] = np.append(dictionary['Qd_max-c2'], np.max(cell_summary['QD'][int(cyc[1]):int(cyc[0])+1]) - cell_summary['QD'][1])
            dictionary['Slope_2to10'] = np.append(dictionary['Slope_2to10'], linreg_2to10.coef_[0][0])
            dictionary['Intercept_2to10'] = np.append(dictionary['Intercept_2to10'], linreg_2to10.intercept_[0])
            dictionary['Slope_6to10'] = np.append(dictionary['Slope_6to10'], linreg_6to10.coef_[0][0])
            dictionary['Intercept_6to10'] = np.append(dictionary['Intercept_6to10'], linreg_6to10.intercept_[0])

            # Other misc. features used only in Full model
            dictionary['Avg_charge_time'] = np.append(dictionary['Avg_charge_time'], np.mean(cell_summary['chargetime'][1:min(6, int(cyc[0])+1)])) #first 5 cycles, 2 to 6
            dictionary['IR_cycle2'] = np.append(dictionary['IR_cycle2'], cell_summary['IR'][1]) # cycle 2
            dictionary['Min_IR'] = np.append(dictionary['Min_IR'], np.min(cell_summary['IR'][ [i + 1 for i in list(np.nonzero(cell_summary['IR'][1:int(cyc[0])])) ] ])) # cycle 2 to 10
            dictionary['IR_cycle'+cycleKey+'-2'] = np.append(dictionary['IR_cycle'+cycleKey+'-2'], cell_summary['IR'][int(cyc[0])] - cell_summary['IR'][1])
            dictionary['MaxT_2to'+cycleKey] = np.append(dictionary['MaxT_2to'+cycleKey], np.max(cell_summary['Tmax'][1:int(cyc[0])+1]))
            dictionary['MinT_2to'+cycleKey] = np.append(dictionary['MinT_2to'+cycleKey], np.min(cell_summary['Tmin'][1:int(cyc[0])+1]))
            dictionary['IntegralT_2to'+cycleKey] = np.append(dictionary['IntegralT_2to'+cycleKey], np.sum([np.trapz(cell_cycles[cyc]['T'], cell_cycles[cyc]['t']) for cyc in map(str, np.arange(1,int(cyc[0])+1,1))]))

            # Append cycle life
            dictionary['cycle_life'] = np.append(dictionary['cycle_life'], int(bat_dict[datatype][batno]['cycle_life']))

            # manual fix: change value of 'b1c18' to 'b1c19'
            if batno == 'b1c19':
                dictionary['IntegralT_2to'+cycleKey][-2] = dictionary['IntegralT_2to'+cycleKey][-1]

    return dictionary