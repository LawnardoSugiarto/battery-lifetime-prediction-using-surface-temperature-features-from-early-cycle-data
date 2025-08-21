import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

"""
This module provides functions for extracting and engineering features from battery cycle data. 
It includes functions to compute statistical summaries, temperature-voltage relationships, and features 
based on Severson's methodology. These features are used for predictive modeling of battery cycle life.

Key Functions:
1. create_T_V: Generates temperature-voltage (T-V) features for charge and discharge phases.
2. create_dTdV: Generates temperature gradient (dT/dV) features for charge and discharge phases.
3. create_Severson: Extracts features based on Severson's methodology, including statistical summaries and linear regression coefficients.
4. get_statistical_summary: Computes statistical metrics (e.g., mean, variance, skewness) for a given data array.

These functions are primarily used to preprocess battery data and extract meaningful features for machine learning models.
"""

def create_T_V(battDict = dict(), cycleList = ['2', '3', '4', '5', '6', '7', '8', '9', '10'], avgUpToCycle = 10):
    """
    Generates temperature-voltage (T-V) features for charge and discharge phases.

    Parameters:
    - battDict (dict): Dictionary containing battery data.
    - cycleList (list): List of cycle numbers to extract features from.
    - avgUpToCycle (int): Maximum cycle number for calculating average features.

    Returns:
    - T_V (dict): Dictionary containing T-V features for charge and discharge phases, 
                  along with statistical summaries and cycle life.
    """
    
    T_V = dict()
    statFeatList = ['Maximum', 'Minimum', 'Max-Min', 'Variance', 'Mean', 'Skewness', 'Kurtosis']
    modeList = ['Charge', 'Discharge']

    # Initialize the dictionary with empty arrays for each feature
    for mode in modeList:
        for statFeat in statFeatList:
            for cycle in cycleList: 
                T_V[mode, statFeat, cycle] = np.array([])
            for cycleEnd in [str(x) for x in range(3, avgUpToCycle+1)]: 
                T_V[mode, statFeat, 'Average_2to'+cycleEnd] = np.array([])

    T_V['cycleLife'] = np.array([])

    # Extract features for each cell and cycle
    for cellNo in battDict.keys():
        cellDict = battDict[cellNo]
        cyclesDict = cellDict['cycles']

        for cycleNo in cycleList:
            cycleDict = cyclesDict[cycleNo]

            # Extract linearized temperature data for charge and discharge phases
            Tc_arr = cycleDict['Tc_lin']
            Td_arr = cycleDict['Td_lin']

            # Compute statistical summaries for charge and discharge
            chFeatDict = get_statistical_summary(Tc_arr, useLog=True)
            disFeatDict = get_statistical_summary(Td_arr, useLog=True)

            # Append features to the dictionary
            for statFeat in statFeatList:
                T_V['Charge', statFeat, cycleNo] = np.append(T_V['Charge', statFeat, cycleNo], chFeatDict[statFeat])
                T_V['Discharge', statFeat, cycleNo] = np.append(T_V['Discharge', statFeat, cycleNo], disFeatDict[statFeat])

        # Append cycle life
        T_V['cycleLife'] = np.append(T_V['cycleLife'], cellDict['cycleLife'])

    # Compute average features across cycles
    for mode in modeList:
        for statFeat in statFeatList:
            for cycleEnd in range(3, avgUpToCycle+1):
                sum = 0
                for cycleAvg in cycleList[:cycleEnd-1]: sum += T_V[mode, statFeat, cycleAvg]
                T_V[mode, statFeat, 'Average_2to'+str(cycleEnd)] = np.append(T_V[mode, statFeat, 'Average_2to'+str(cycleEnd)], sum/(cycleEnd-1))

    return T_V

def create_dTdV(battDict = dict(), cycleList = ['2', '3', '4', '5', '6', '7', '8', '9', '10'], avgUpToCycle = 10):
    """
    Generates temperature gradient (dT/dV) features for charge and discharge phases.

    Parameters:
    - battDict (dict): Dictionary containing battery data.
    - cycleList (list): List of cycle numbers to extract features from.
    - avgUpToCycle (int): Maximum cycle number for calculating average features.

    Returns:
    - dTdV (dict): Dictionary containing dT/dV features for charge and discharge phases, 
                   along with statistical summaries and cycle life.
    """
    
    dTdV = dict()
    statFeatList = ['Maximum', 'Minimum', 'Max-Min', 'Variance', 'Mean', 'Skewness', 'Kurtosis']
    modeList = ['Charge', 'Discharge']

    # Initialize the dictionary with empty arrays for each feature
    for mode in modeList:
        for statFeat in statFeatList:
            for cycle in cycleList: dTdV[mode, statFeat, cycle] = np.array([])
            for cycleEnd in [str(x) for x in range(3, avgUpToCycle+1)]: dTdV[mode, statFeat, 'Average_2to'+cycleEnd] = np.array([])
    
    dTdV['cycleLife'] = np.array([])

    # Extract features for each cell and cycle
    for cellNo in battDict.keys():
        cellDict = battDict[cellNo]
        cyclesDict = cellDict['cycles']

        for cycleNo in cycleList:
            cycleDict = cyclesDict[cycleNo]

            # Compute dT/dV for charge and discharge phases
            dTdVc_arr = np.diff(cycleDict['Tc_lin']) / np.diff(cycleDict['Vc_lin'])
            dTdVd_arr = np.diff(cycleDict['Td_lin']) / np.diff(cycleDict['Vd_lin'])

            # Compute statistical summaries for charge and discharge
            chFeatDict = get_statistical_summary(dTdVc_arr, useLog=True)
            disFeatDict = get_statistical_summary(dTdVd_arr, useLog=True)

            # Append features to the dictionary
            for statFeat in statFeatList:
                dTdV['Charge', statFeat, cycleNo] = np.append(dTdV['Charge', statFeat, cycleNo], chFeatDict[statFeat])
                dTdV['Discharge', statFeat, cycleNo] = np.append(dTdV['Discharge', statFeat, cycleNo], disFeatDict[statFeat])
        # Append cycle life
        dTdV['cycleLife'] = np.append(dTdV['cycleLife'], cellDict['cycleLife'])

    # Compute average features across cycles
    for mode in modeList:
        for statFeat in statFeatList:
            for cycleEnd in range(3, avgUpToCycle+1):
                sum = 0
                for cycleAvg in cycleList[:cycleEnd-1]: sum += dTdV[mode, statFeat, cycleAvg]
                dTdV[mode, statFeat, 'Average_2to'+str(cycleEnd)] = np.append(dTdV[mode, statFeat, 'Average_2to'+str(cycleEnd)], sum/(cycleEnd-1))

    return dTdV

def create_Severson(battDict = dict(), cycleList = ['2', '10']):
    """
    Extracts features based on Severson's methodology, including statistical summaries and linear regression coefficients.

    Parameters:
    - battDict (dict): Dictionary containing battery data.
    - cycleList (list): List of cycle numbers to extract features from.

    Returns:
    - Severson (dict): Dictionary containing Severson features, including Q(V) statistics, 
                       charge/discharge summaries, and other derived metrics.
    """
    
    Severson = dict()
    cycNo = cycleList[1]
    featNameList = ['Q_V_Variance', 'Q_V_Mean', 'Q_V_Skewness', 'Q_V_Kurtosis', 'Q_V_Minimum', 'Q_V_2_5V',
                    'Qd_c2', 'Qd_c'+cycNo, 'Qd_max-c2', 'Slope_2to'+cycNo, 'Intercept_2to'+cycNo,
                    'Avg_charge_time', 'IR_cycle2', 'Min_IR', 'IR_cycle'+cycNo+'-2', 'MaxT_2to'+cycNo, 'MinT_2to'+cycNo, 'IntegralT_2to'+cycNo,
                    'cycleLife']
    
    # Initialize the dictionary with empty arrays for each feature
    for featName in featNameList:
        Severson[featName] = np.array([])

    # Extract features for each cell
    for cellNo in battDict.keys():
        cellDict = battDict[cellNo]
        cyclesDict = cellDict['cycles']
        summaryDict = cellDict['summary']

        dQ = cyclesDict[cycleList[1]]['Qd_lin'] - cyclesDict[cycleList[0]]['Qd_lin']

        # Compute Q(V) statistics
        Severson['Q_V_Variance'] = np.append(Severson['Q_V_Variance'], np.log10(np.abs(np.var(dQ))))
        Severson['Q_V_Mean'] = np.append(Severson['Q_V_Mean'], np.log10(np.abs(np.mean(dQ))))
        Severson['Q_V_Skewness'] = np.append(Severson['Q_V_Skewness'], np.log10(np.abs(skew(dQ))))
        Severson['Q_V_Kurtosis'] = np.append(Severson['Q_V_Kurtosis'], np.log10(np.abs(kurtosis(dQ))))
        Severson['Q_V_Minimum'] = np.append(Severson['Q_V_Minimum'], np.log10(np.abs(min(dQ[np.nonzero(dQ)[0]]))))
        Severson['Q_V_2_5V'] = np.append(Severson['Q_V_2_5V'], np.log10(np.abs(dQ[-1])))
        
        # Compute charge/discharge summaries and linear regression coefficients
        cycleNoList = summaryDict['cycle'][0:int(cycleList[1])+1].reshape(-1,1)
        summaryQd = summaryDict['Qd'][0:int(cycleList[1])+1].reshape(-1,1)
        linreg_2toX =  LinearRegression().fit(cycleNoList[2:], summaryQd[2:])

        Severson['Qd_c2'] = np.append(Severson['Qd_c2'], summaryDict['Qd'][2])
        Severson['Qd_c'+cycNo] = np.append(Severson['Qd_c'+cycNo], summaryDict['Qd'][int(cycNo)])
        Severson['Qd_max-c2'] = np.append(Severson['Qd_max-c2'], np.max(summaryDict['Qd'][2:int(cycNo)+1]) - summaryDict['Qd'][2])
        Severson['Slope_2to'+cycNo] = np.append(Severson['Slope_2to'+cycNo], linreg_2toX.coef_[0][0])
        Severson['Intercept_2to'+cycNo] = np.append(Severson['Intercept_2to'+cycNo], linreg_2toX.intercept_[0])

        # Compute other misc. features
        Severson['Avg_charge_time'] = np.append(Severson['Avg_charge_time'], np.mean(summaryDict['chargetime'][2:min(7, int(cycleList[1])+1)])) #first 5 cycles, 2 to 6
        Severson['IR_cycle2'] = np.append(Severson['IR_cycle2'], summaryDict['IRavg'][2]) # cycle 2
        Severson['Min_IR'] = np.append(Severson['Min_IR'], np.min(summaryDict['IRavg'][ [i for i in list(np.nonzero(summaryDict['IRavg'][2:int(cycleList[1])+1])[0]+2) ] ])) # cycle 2 to X
        Severson['IR_cycle'+cycNo+'-2'] = np.append(Severson['IR_cycle'+cycNo+'-2'], summaryDict['IRavg'][int(cycleList[1])] - summaryDict['IRavg'][2])
        Severson['MaxT_2to'+cycNo] = np.append(Severson['MaxT_2to'+cycNo], np.max(summaryDict['Tmax'][2:int(cycleList[1])+1]))
        Severson['MinT_2to'+cycNo] = np.append(Severson['MinT_2to'+cycNo], np.min(summaryDict['Tmin'][2:int(cycleList[1])+1]))
        Severson['IntegralT_2to'+cycNo] = np.append(Severson['IntegralT_2to'+cycNo], np.sum([np.trapz(cyclesDict[cyc]['T_cell'], cyclesDict[cyc]['t']) for cyc in map(str, np.arange(2,int(cycleList[1])+1,1))]))

        Severson['cycleLife'] = np.append(Severson['cycleLife'], int(cellDict['cycleLife']))

    return Severson

def get_statistical_summary(dataArr = np.ndarray, useLog = False):
    """
    Computes statistical metrics (e.g., mean, variance, skewness) for a given data array.

    Parameters:
    - dataArr (np.ndarray): Input data array.
    - useLog (bool): If True, applies a log10 transformation to the computed metrics.

    Returns:
    - returnDict (dict): Dictionary containing statistical metrics.
    """
    returnDict = {}
    returnDict['Kurtosis'] = kurtosis(dataArr)
    returnDict['Maximum'] = np.max(dataArr)
    returnDict['Mean'] = np.mean(dataArr)
    returnDict['Minimum'] = np.min(dataArr[np.nonzero(dataArr)])
    returnDict['Skewness'] = skew(dataArr)
    returnDict['Variance'] = np.var(dataArr)
    returnDict['Max-Min'] = returnDict['Maximum'] - returnDict['Minimum']

    if useLog: 
        for keyName in returnDict.keys(): 
            returnDict[keyName] = np.log10(np.abs(returnDict[keyName]))

    return returnDict