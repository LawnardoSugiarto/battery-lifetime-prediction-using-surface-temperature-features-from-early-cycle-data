import sklearn.feature_selection
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from function_utils import root_mean_squared_percentage_error

import math

"""
This module provides utility functions for training and evaluating machine learning models on battery data.
It includes methods for calculating performance metrics such as R², MAE, MAPE, RMSE, and RMSPE, and for
returning formatted output strings summarizing model performance.
"""
# To calculate ROOT mean_squared_error, do this
# mean_squared_error(y_true, y_pred, squared=False)


def feature_target_score(X, y, featureKeynameList, printTitle='ALL CELLS', toPrint=True, f_reg=False):
    """
    Calculates the Pearson's r correlation coefficient for each feature in the feature matrix (`X`) with respect to the 
    target variable (`y`). Optionally calculates F-statistics and p-values for each feature using `f_regression`. 

    Inputs:
    - X: Feature matrix (numpy array) where each column represents a feature.
    - y: Target variable (numpy array).
    - featureKeynameList: List of feature names corresponding to the columns of `X`.
    - printTitle: Title to display when printing results (default: 'ALL CELLS').
    - toPrint: Boolean flag to print the results (default: True).
    - f_reg: Boolean flag to return and print F-statistics and p-values (default: False).

    Outputs:
    - rScore: Array of Pearson's r correlation coefficients for each feature.
    - fScore: Tuple containing F-statistics and p-values for each feature (returned only if `f_reg` is True).
    """
    # Calculate Pearson's r correlation coefficient
    rScore = sklearn.feature_selection.r_regression(X, y)

    if(toPrint):
        print("---- %s ----" % printTitle)
        print("Pearson's r correlation coefficient")
        for i in range(len(rScore)):
            print(featureKeynameList[i], " has a Pearson's r of ", rScore[i])
        
    if f_reg:
        print("\nF-statistic and p-values")
        fScore = sklearn.feature_selection.f_regression(X, y)
        for i in range(len(fScore[0])):
            print(featureKeynameList[i], " has F-statistic ", fScore[0][i], " and p-values of ", fScore[1][i])
        
        return rScore, fScore
        
    else: 
        return rScore

def print_result(cycleNo, model, X, y, featurelst, idlst = [41, 83, 123], r2AllOnly = 0, returnR2 = 0):
    """
    Prints and returns a formatted string summarizing the performance of a trained regression model. It includes coefficients,
    intercept, R² score, and metrics such as MAE, MAPE, and RMSE for training, test, and secondary test datasets. Optionally,
    returns the R² score for all data concenated.

    Inputs:
    - cycleNo: Cycle number (used for labeling results; default: 0).
    - model: Trained regression model (e.g., sklearn's LinearRegression or similar).
    - X: Feature matrix (numpy array) used for predictions.
    - y: True target values (numpy array).
    - featurelst: List of feature names corresponding to the columns of `X`.
    - idlst: List containing indices (length) for splitting the data into training, test, and secondary test datasets.
    - r2AllOnly: Flag to print only the R² score for all data (default: 0).
    - returnR2: Flag to return the R² score for all data instead of the formatted string (default: 0).

    Outputs:
    - stringOutput: Formatted string summarizing model performance metrics (if returnR2 == 0).
    - R² score: R² score for all data (if returnR2 != 0).
    """
    stringOutput = ''
    if(cycleNo!=0): stringOutput += "Cycle no: " + cycleNo +"\nCoefficient:\n"

    # Print coefficients for each feature
    if(len(model.coef_)==1 & len(featurelst) > 1):
        for f, co in zip(featurelst ,model.coef_[0]):
            stringOutput += str(f) + " : " + str(co) + "\n"
    else:
        for f, co in zip(featurelst ,model.coef_):
            stringOutput += str(f) + " : " + str(co) + "\n"

    # Print metrics if r2AllOnly flag is not set
    if r2AllOnly == 0:
        stringOutput += ("\nIntercept: " + str(model.intercept_) + "\nr2-score: %.3f" % (model.score(X, y)) + "\nMAE_all: %.1f" % (mean_absolute_error(y_true=10**y, y_pred=10**model.predict(X))) + 
                        "\nMAE_train: %.1f" % (mean_absolute_error(y_true=10**y[0:idlst[0]], y_pred=10**model.predict(X[0:idlst[0]]))) + 
                        "\nMAE_test: %.1f" % (mean_absolute_error(y_true=10**y[idlst[0]:idlst[1]], y_pred=10**model.predict(X[idlst[0]:idlst[1]]))) + 
                        "\nMAE_2ndTest: %.1f" % (mean_absolute_error(y_true=10**y[idlst[1]:], y_pred=10**model.predict(X[idlst[1]:]))) + 
                        "\nMAPE_all: %.1f %%" % (mean_absolute_percentage_error(y_true=10**y, y_pred=10**model.predict(X)) * 100) + 
                        "\nMAPE_train: %.1f %%" % (mean_absolute_percentage_error(y_true=10**y[0:idlst[0]], y_pred=10**model.predict(X[0:idlst[0]])) * 100) + 
                        "\nMAPE_test: %.1f %%" % (mean_absolute_percentage_error(y_true=10**y[idlst[0]:idlst[1]], y_pred=10**model.predict(X[idlst[0]:idlst[1]])) * 100) + 
                        "\nMAPE_2ndTest: %.1f %%" % (mean_absolute_percentage_error(y_true=10**y[idlst[1]:], y_pred=10**model.predict(X[idlst[1]:])) * 100) + 
                        "\nRMSE_all: %.1f" % (math.sqrt(mean_squared_error(y_true=10**y, y_pred=10**model.predict(X)))) + 
                        "\nRMSE_train: %.1f" % (math.sqrt(mean_squared_error(y_true=10**y[0:idlst[0]], y_pred=10**model.predict(X[0:idlst[0]])))) + 
                        "\nRMSE_test: %.1f" % (math.sqrt(mean_squared_error(y_true=10**y[idlst[0]:idlst[1]], y_pred=10**model.predict(X[idlst[0]:idlst[1]])))) + 
                        "\nRMSE_2ndTest: %.1f" % (math.sqrt(mean_squared_error(y_true=10**y[idlst[1]:], y_pred=10**model.predict(X[idlst[1]:])))) + 
                        "\nRMSPE_all: %.1f %%" % (root_mean_squared_percentage_error(y_true=10**y, y_pred=10**model.predict(X)) * 100) + 
                        "\nRMSPE_train: %.1f %%" % (root_mean_squared_percentage_error(y_true=10**y[0:idlst[0]], y_pred=10**model.predict(X[0:idlst[0]])) * 100) + 
                        "\nRMSPE_test: %.1f %%" % (root_mean_squared_percentage_error(y_true=10**y[idlst[0]:idlst[1]], y_pred=10**model.predict(X[idlst[0]:idlst[1]])) * 100) + 
                        "\nRMSPE_2ndTest: %.1f %%" % (root_mean_squared_percentage_error(y_true=10**y[idlst[1]:], y_pred=10**model.predict(X[idlst[1]:])) * 100) ) 
        
        stringOutput += (   "\nLog-scale scores" + "\nAll data r2-score: %.3f" % (model.score(X=X, y=y)) + "\nTrain data r2-score: %.3f" % (model.score(X=X[0:idlst[0]], y=y[0:idlst[0]])) + 
                        "\nTest data r2-score: %.3f" % (model.score(X=X[idlst[0]:idlst[1]], y=y[idlst[0]:idlst[1]])) + 
                        "\nSecondary test data r2-score: %.3f" % (model.score(X=X[idlst[1]:], y=y[idlst[1]:]))   )
        
        print(stringOutput)

    # Print only R² score if r2AllOnly flag is set
    else:
        print("All data score: %.3f" % (model.score(X =X, y = y)))
        
    # Optionally return the R² score for all data
    if (returnR2 != 0):
        return model.score(X = X, y= y)
    
    return stringOutput