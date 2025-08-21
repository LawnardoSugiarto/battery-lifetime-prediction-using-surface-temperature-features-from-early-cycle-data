import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math

"""
This module provides utility functions for visualizing battery cycle data and model performance. 

Key Functions:
1. plot_statistical_summary: Creates scatter plots of statistical features against the target variable (e.g., cycle life).
2. build_all_comparison_graph: Visualizes true vs. predicted values for regression models, with residual error histograms.

These functions are primarily used for exploratory data analysis and evaluating the performance of machine learning models.
"""

def plot_statistical_summary(featNameList = list(), figSize = (18,12), featDict=dict(), yArr=np.ndarray, 
                             textSize=20, fontName='Microsoft Sans Serif', markerSize=100, lineWidth=3,
                             cmapcolor="coolwarm", fillcolor='white'):
    """
    Creates scatter plots of statistical features against the target variable (e.g., cycle life).

    Parameters:
    - featNameList (list): List of feature names to plot.
    - figSize (tuple): Size of the figure (width, height).
    - featDict (dict): Dictionary containing feature values for each feature name.
    - yArr (np.ndarray): Target variable (e.g., cycle life).
    - textSize (int): Font size for plot titles and labels.
    - fontName (str): Font name for text in the plots.
    - markerSize (int): Size of the scatter plot markers.
    - lineWidth (int): Width of the marker edges.
    - cmapcolor (str): Colormap for coloring the markers.
    - fillcolor (str): Fill color for the markers.

    Returns:
    - None: Displays the scatter plots.
    """
    cmap = plt.get_cmap(cmapcolor)
    scale_cmap = lambda x : (x - 400) / (800 - 400) # Normalize the colormap scale

    # Create a grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=figSize)

    # Iterate through each feature and create a scatter plot
    for axNo, featName in enumerate(featNameList):
        pearson = np.corrcoef(featDict[featName], yArr) # Calculate Pearson correlation
        ax = axs[axNo//3, axNo%3] # Select subplot
        ax.set_title(f"{featName}, ρ={pearson[0,1]:.3f}", fontsize=textSize, fontname=fontName)
        ax.scatter(featDict[featName], yArr, s=markerSize, c=fillcolor, linewidths=lineWidth, 
                   edgecolors=cmap(scale_cmap(yArr))) # Scatter plot with colored edges
        
def build_all_comparison_graph(X, y, regression, len_list, name, markerArr = False, yScaler = 0, textSize=64, fontName='Microsoft Sans Serif', markerSize=1800, save=100):
    """
    Visualizes true vs. predicted values for regression models, with residual error histograms.

    Parameters:
    - X (np.ndarray): Feature matrix used for predictions.
    - y (np.ndarray): True target values.
    - regression (object): Trained regression model.
    - len_list (list): List containing the lengths of train, test, and optional secondary test sets.
    - name (str): Name of the model or feature set being visualized.
    - markerArr (np.ndarray): Array of marker styles for different data points.
    - yScaler (object): Scaler object for inverse transforming scaled target values (optional).
    - textSize (int): Font size for plot titles and labels.
    - fontName (str): Font name for text in the plots.
    - markerSize (int): Size of the scatter plot markers.
    - save (str or int): File path to save the plot, or 100 to skip saving.

    Returns:
    - stringOutput (str): Summary of R² scores for train, test, and secondary test sets.
    """
    y_all = y
    y_all_pred = regression.predict(X).reshape(-1, 1)

    # Default marker array if not provided
    if markerArr == False: 
        markerArr = np.array(['o']*len(y))
    
    # Inverse transform target values if a scaler is provided
    if yScaler != 0:
        y_all = yScaler.inverse_transform(y_all)
        y_all_pred = yScaler.inverse_transform(y_all_pred.reshape(-1, 1))

    # Split data into train, test, and optional secondary test sets
    y_train = y_all[0:len_list[0]]
    y_test = y_all[len_list[0]:(len_list[0]+len_list[1])]
    y_train_pred = y_all_pred[0:len_list[0]]
    y_test_pred = y_all_pred[len_list[0]:(len_list[0]+len_list[1])]

    # Create the main scatter plot
    fig, ax = plt.subplots(figsize=(15,15))

    # Plot train data
    for i, idx in enumerate(range(0, len_list[0])):
        ax.scatter(y_train[i], y_train_pred[i], color='forestgreen', label='Train', s=markerSize, 
                   marker=markerArr[idx], edgecolors='k', linewidths=5, zorder=1)
        
    # Plot test data
    for i, idx in enumerate(range(len_list[0], len_list[0]+len_list[1])):
        ax.scatter(y_test[i], y_test_pred[i], color='darkorange', label='Primary test', s=markerSize, 
                   marker=markerArr[idx], edgecolors='k', linewidths=5, zorder=2)
        
    # Add a diagonal reference line
    ax.plot([100, 500], [100, 500], color='black', linestyle='--', linewidth=4, zorder=2)
    
    # Set axis labels and ticks
    ax.set_ylabel("Estimated cycle life", fontsize=textSize+5, fontname=fontName)
    ax.set_xlabel("True cycle life",fontsize=textSize+5, fontname=fontName)
    ax.set_xticks([100, 200, 300, 400])
    ax.set_xticklabels([100, 200, 300, 400], fontname=fontName, y = -0.02)
    ax.set_yticks([200, 300, 400])
    ax.set_yticklabels([200, 300, 400], fontname=fontName, x = -0.02)
    ax.set_xlim([100, 500])
    ax.set_ylim([100, 500])

    # Add r2-score to the plot
    ax.text(x=0.6, y=0.12, s="$R^{}$={:.3f}".format('2', r2_score(y_true = y_all, y_pred = y_all_pred)), 
            fontsize=textSize, zorder=3, fontname=fontName, transform=ax.transAxes)
 
    # Configure tick parameters
    ax.tick_params(labelsize=textSize, left=False, bottom=False)
    ax.yaxis.set_label_coords(-0.23, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.13)
    plt.grid()

    # Plot for inset histogram plot inside the main figure
    inset_ax = inset_axes(ax, width="38%", height="32%",
                          loc="upper left", bbox_to_anchor = (0.09, -0.07, 1, 1), bbox_transform=ax.transAxes)
    inset_ax.set_facecolor('none')
    residual = (y_all - y_all_pred).reshape(-1)
    counts, bins, bars = inset_ax.hist(residual, bins=10, range=(-150, 150),
                  color='dimgray', edgecolor='white')
                  
    # Configure inset axis ticks and labels
    maxCount = math.ceil(counts.max() / 5)
    countTicks = np.arange(0, maxCount*5+1, 5)
    inset_ax.set_xticks([-150, -75, 0, 75, 150])
    inset_ax.set_xticklabels([-150, -75, 0, 75, 150], fontname=fontName, y=-0.03)
    inset_ax.set_yticks(countTicks)
    inset_ax.set_yticklabels(countTicks, fontname=fontName, x = -0.03)
    inset_ax.tick_params(labelsize=textSize/11*6)
    inset_ax.yaxis.set_tick_params(direction='in', length=15, which='major', width=4)
    inset_ax.xaxis.set_tick_params(direction='out', length=10, which='major', width=4)
    
    inset_ax.set_title("Residual error (cycle)", fontsize=textSize/11*7, x=0.5, y=1.03)

    # Configure inset axis borders
    for axis in ['top', 'bottom', 'left', 'right']: 
        ax.spines[axis].set_linewidth(4)
        inset_ax.spines[axis].set_linewidth(4)

    # Print r2-scores
    stringOutput = ("\nNormal-scale scores" +
                    "\nAll data score: %.3f" % (r2_score(y_true=y_all, y_pred=y_all_pred)) +
                    "\nTrain data score: %.3f" % (r2_score(y_true=y_train, y_pred=y_train_pred)) +
                    "\nTest data score: %.3f" % (r2_score(y_true=y_test, y_pred=y_test_pred))
                    )

    # Handle secondary test set if provided                
    if len(len_list) > 2:
        y_sectest = y_all[(len_list[0]+len_list[1]):]
        y_sectest_pred = y_all_pred[(len_list[0]+len_list[1]):]
        for i, idx in enumerate(range(len_list[0]+len_list[1], len(y_all))):
            ax.scatter(y_sectest[i], y_sectest_pred[i], color='darkorange', label='Secondary test', s=markerSize, marker=markerArr[idx], zorder=3)
        stringOutput += ("\nSecondary test data score: %.3f" % (r2_score(y_true=y_sectest, y_pred=y_sectest_pred)))

    print(stringOutput)
    
    # Save the plot if a file path is provided
    if(save!=100):
        fig.savefig(save, dpi=1000, bbox_inches='tight')
    plt.show()
    
    return stringOutput