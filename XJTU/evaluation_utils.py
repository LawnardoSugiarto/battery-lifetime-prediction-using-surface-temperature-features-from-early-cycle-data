import numpy as np
import sklearn.metrics, scipy.stats, sklearn.feature_selection

"""
This module provides utility functions for evaluating the performance of machine learning models 
and analyzing relationships between variables. It includes functions for calculating error metrics, 
correlation coefficients, and feature importance scores.

Key Functions:
1. root_mean_squared_percentage_error: Computes the root mean squared percentage error (RMSPE) between true and predicted values.
2. error_metrics: Returns a collection of common error metrics, including RMSE, RMSPE, MAE, MAPE, and R² score.
3. correlation_metrics: Calculates the Pearson correlation coefficient between two variables.
4. rScore: Computes the Pearson correlation coefficient (r-score) for each feature in a dataset.

These functions are primarily used to evaluate model performance, assess feature relevance, and analyze 
relationships between variables in battery cycle life prediction tasks or other regression problems.
"""

def root_mean_squared_percentage_error(y_true = float, y_pred = float):
    return np.sqrt(np.mean(np.square (( (y_true - y_pred) / y_true ) ), axis = 0))

def error_metrics(y_true = float, y_pred = float):
    return (sklearn.metrics.root_mean_squared_error(y_true = y_true, y_pred = y_pred),
            root_mean_squared_percentage_error(y_true = y_true, y_pred = y_pred),
            sklearn.metrics.mean_absolute_error(y_true = y_true, y_pred = y_pred),
            sklearn.metrics.mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred),
            sklearn.metrics.r2_score(y_true = y_true, y_pred = y_pred))

def correlation_metrics(var1 = float, var2 = float):
    return (scipy.stats.pearsonr(x = var1, y = var2))

def rScore (X, y, featureKeynameList, printTitle = 'All cells', toPrint=False):
    rScore = sklearn.feature_selection.r_regression(X, y)
    
    if(toPrint):
        print("----- %s -----" % printTitle)
        print("Pearson's r correlation coefficient")
        for i in range(len(rScore)): print(featureKeynameList[i], " has rScore of ", rScore[i])
        
    return rScore

