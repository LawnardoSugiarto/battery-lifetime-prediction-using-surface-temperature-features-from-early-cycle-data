import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from evaluation_utils import root_mean_squared_percentage_error

"""
This module provides utility functions for training and evaluating machine learning models on battery data.
It includes methods for calculating performance metrics such as R², MAE, MAPE, RMSE, and RMSPE, and for
returning formatted output strings summarizing model performance.
"""

def print_result(cycleNo, model, X, y, featurelst, idlst = [32, 48, 64],r2AllOnly = 0, returnR2 = 0, yScaler = 0):
    """
    Evaluates and prints regression model performance across training, testing, and 
    optional secondary test sets.
    
    Parameters:
    - cycleNo (int): Optional cycle number for display.
    - model: Trained regression model with .predict and .coef_ attributes.
    - X (array): Feature matrix.
    - y (array): Target values.
    - featurelst (list): List of feature names for coefficient display.
    - idlst (list): Index boundaries for train/test/secondary test splits [train_end, test_end, second_test_end].
    - r2AllOnly (bool): If set to 1, only prints R² score for all data.
    - returnR2 (bool): If set to 1, returns R² score instead of full output string.
    - yScaler (object): Optional scaler for inverse-transforming predictions and targets.
    
    Returns:
    - stringOutput (str): Formatted performance metrics (unless returnR2 is set).
    """

    stringOutput = ''
    
    # Display cycle number and model coefficients
    if(cycleNo!=0): 
        stringOutput += "Cycle no: " + cycleNo +"\nCoefficient:\n"

    # Handle coefficient formatting for multi-output models
    if(len(model.coef_)==1 & len(featurelst) > 1):
        for f, co in zip(featurelst ,model.coef_[0]):
            stringOutput += str(f) + " : " + str(co) + "\n"
    else:
        for f, co in zip(featurelst ,model.coef_):
            stringOutput += str(f) + " : " + str(co) + "\n"

    # Predict and optionally inverse-transform predictions
    y_all = y
    y_all_pred = model.predict(X)
    if yScaler != 0:
        y_all = yScaler.inverse_transform(y_all)
        y_all_pred = yScaler.inverse_transform(y_all_pred.reshape(-1, 1))
    
    # Split data into train, test, and optional second test sets
    y_train = y_all[0:idlst[0]]
    y_test = y_all[idlst[0]:idlst[1]]
    y_train_pred = y_all_pred[0:idlst[0]]
    y_test_pred = y_all_pred[idlst[0]:idlst[1]]
    
    if r2AllOnly == 0:
        # Print intercept and error metrics for all splits
        stringOutput += ("\nIntercept: " + str(model.intercept_) + 
                         "\nMAE_all: %.1f" % (mean_absolute_error(y_true=y_all, y_pred=y_all_pred)) + 
                        "\nMAPE_all: %.1f %%" % (mean_absolute_percentage_error(y_true=y_all, y_pred=y_all_pred) * 100) + 
                        "\nRMSE_all: %.1f" % (math.sqrt(mean_squared_error(y_true=y_all, y_pred=y_all_pred))) + 
                        "\nRMSPE_all: %.1f %%" % (root_mean_squared_percentage_error(y_true=y_all, y_pred=y_all_pred) * 100)  +
                         
                        "\nMAE_train: %.1f" % (mean_absolute_error(y_true=y_train, y_pred=y_train_pred)) + 
                        "\nMAPE_train: %.1f %%" % (mean_absolute_percentage_error(y_true=y_train, y_pred=y_train_pred) * 100) + 
                        "\nRMSE_train: %.1f" % (math.sqrt(mean_squared_error(y_true=y_train, y_pred=y_train_pred))) + 
                        "\nRMSPE_train: %.1f %%" % (root_mean_squared_percentage_error(y_true=y_train, y_pred=y_train_pred) * 100) + 

                        "\nMAE_test: %.1f" % (mean_absolute_error(y_true=y_test, y_pred=y_test_pred)) + 
                        "\nMAPE_test: %.1f %%" % (mean_absolute_percentage_error(y_true=y_test, y_pred=y_test_pred) * 100) + 
                        "\nRMSE_test: %.1f" % (math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))) + 
                        "\nRMSPE_test: %.1f %%" % (root_mean_squared_percentage_error(y_true=y_test, y_pred=y_test_pred) * 100) 
                        )
        
        
        # If secondary test set exists, evaluate it                
        if len(idlst) > 2:  
            y_second_test = y_all[idlst[1]:]
            y_second_test_pred = y_all_pred[idlst[1]:]
            stringOutput += ("\nMAE_2ndTest: %.1f" % (mean_absolute_error(y_true=y_second_test, y_pred=y_second_test_pred)) + 
                            "\nMAPE_2ndTest: %.1f %%" % (mean_absolute_percentage_error(y_true=y_second_test, y_pred=y_second_test_pred) * 100) + 
                            "\nRMSE_2ndTest: %.1f" % (math.sqrt(mean_squared_error(y_true=y_second_test, y_pred=y_second_test_pred))) + 
                            "\nRMSPE_2ndTest: %.1f %%" % (root_mean_squared_percentage_error(y_true=y_second_test, y_pred=y_second_test_pred) * 100)
                            )

        # Print r2-scores for each split
        stringOutput += ("\nR2 normal-scale scores" + 
                         "\nAll data r2-score: %.3f" % (r2_score(y_true=y_all, y_pred=y_all_pred)) + 
                         "\nTrain data r2-score: %.3f" % (r2_score(y_true=y_train, y_pred=y_train_pred)) + 
                        "\nTest data r2-score: %.3f" % (r2_score(y_true=y_test, y_pred=y_test_pred)) 
                        ) 
        if len(idlst) > 2:  stringOutput += ("\nSecondary test data r2-score: %.3f" % (r2_score(y_true=y_second_test, y_pred=y_second_test_pred))   )
        
        print(stringOutput)
        
    else:
        # Print only R² score for all data
        print("All data score: %.3f" % (r2_score(y_true=y_all, y_pred=y_all_pred)))
        
    # Optionally return R² score
    if (returnR2 != 0):
        return r2_score(y_true=y_all, y_pred=y_all_pred)
    
    return stringOutput