import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
This module provides utility functions for visualizing battery cycling data. It includes methods for plotting
step sections of battery cycles, showing temperature, current, voltage, and capacity across different steps.
The plots are organized into subplots for easier analysis and comparison.

NOTE: Most functions are removed from this module, as they are not used in the current context.
"""


# Show the rest, charge, rest after charge and discharge steps into four different columns; with two rows of Temperature & Current and Voltage & Capacity plots
def plot_step_sections(cell_dict, cycles_lst, annotate=True, foldername=None):
    """
    Plots step sections of battery cycles in a grid layout with 2 rows and 4 columns (total 8 subplots).
    The top row shows temperature and current, while the bottom row shows voltage and capacity for each step.
    Each column corresponds to a specific step: rest before charge, charge, rest after charge, and discharge.

    Inputs:
    - cell_dict: Dictionary containing battery data for a specific cell (e.g., `bat_dict['NMC_data']['2']`).
    - cycles_lst: List of cycle numbers to plot.
    - annotate: Boolean flag to annotate plots with time taken for each step and CV hold (default: True).
    - foldername: String specifying the folder name to save the plots (default: None).

    Outputs:
    - Saves the plots to the specified folder if `foldername` is provided.
    """

    for cycle in cycles_lst:
        cell = cell_dict['cycles'][cycle]
        # Choose the step index using function_utils.determine_step_indices() for our NMC data
        # step_id = function_utils.determine_step_indices(cell_dict, cycle)
        step_id = [3, 4, 5, 6] # Hardcoded step indices for demonstration
        
        # Create the subplots
        # Extract indices for each step
        idx_1 = np.where(cell['Step_index'] == step_id[0])[0][0: np.where(cell['Step_index'] == step_id[1])[0][0]]
        idx_2 = np.where(cell['Step_index']==step_id[1])[0]
        idx_3 = np.where(cell['Step_index'] == step_id[0])[0][np.where(cell['Step_index'] == step_id[1])[0][0]:]
        idx_4 = np.where(cell['Step_index']==step_id[2])[0]
        
        # Calculate width ratios for subplots based on time taken for each step
        t_final = cell['t'][idx_4[-1]]
        r1 = (cell['t'][idx_1[-1]] - cell['t'][idx_1[0]]) / t_final + 0.3
        r2 = (cell['t'][idx_2[-1]] - cell['t'][idx_2[0]]) / t_final
        r3 = (cell['t'][idx_3[-1]] - cell['t'][idx_3[0]]) / t_final - 0.3
        r4 = (cell['t'][idx_4[-1]] - cell['t'][idx_4[0]]) / t_final

        # Create subplots with calculated width ratios
        fig, axs = plt.subplots(2,4,figsize=(20,10), 
                                gridspec_kw={'width_ratios': [r1,r2,r3,r4]})
        axs = axs.ravel()
        fig.suptitle('Cycle no. {} for cell with protocol {}'.format(cycle, cell_dict['charge_policy']), fontsize=18)
        
        # For each subplot column
        for i, idx in enumerate([idx_1, idx_2, idx_3, idx_4]):
            # Create a new twin, sharing axes
            axs_top = axs[i].twinx()
            axs_bot = axs[i+4].twinx()

            # Plot for temperature and current
            axs[i].plot(cell['t'][idx], cell['T_cell'][idx], color = 'k')
            axs[i].set_ylim([np.min(cell['T_cell']), np.max(cell['T_cell'])+0.5])
            axs[i].set_xlabel('Time (s)', fontsize=14)
            axs[i].grid()
            axs_top.plot(cell['t'][idx], cell['I'][idx], color = 'r')
            axs_top.set_ylim([np.min(cell['I']), np.max(cell['I'])+0.2])
            

            # Plot for voltage and capacity
            axs[i+4].plot(cell['t'][idx], cell['V'][idx], color = 'orange')
            axs[i+4].set_ylim([np.min(cell['V']), np.max(cell['V'])+0.1])
            axs[i+4].grid()
            axs_bot.plot(cell['t'][idx], cell['Qc'][idx] - cell['Qd'][idx], color = 'g')
            axs_bot.set_ylim([np.min(cell['Qc']), np.max(cell['Qc'])+0.1])
            
            # Write some text with time information
            if(annotate==True):
                if(len(idx)<2): 
                    time = 0
                else:
                    time = (cell['t'][idx[-1]] - cell['t'][idx[0]]) * 60
                # Add text to the top subplot    
                axs[i].text(0.1, 0.9, 'Time: '+str(int(time))+ ' s', transform = axs[i].transAxes, bbox=dict(facecolor='blue', alpha=0.3, linestyle='--'),
                fontsize='large')


        # Label the axes
        axs[0].set_ylabel('Temperature', fontsize=14, color = 'k')
        axs[4].set_ylabel('Voltage', fontsize=14, color = 'orange')
        axs_top.set_ylabel('Current', fontsize=14, color = 'r')
        axs_bot.set_ylabel('Capacity (Ah)', fontsize=14, color = 'g')
        axs[4].set_xlabel('Charge (CC)', fontsize=14)
        axs[5].set_xlabel('Rest (CC Charge)', fontsize=14)
        axs[6].set_xlabel('Charge (CV)', fontsize=14)
        axs[7].set_xlabel('Discharge (CCCV)', fontsize=14)
        
        fig.tight_layout()

        # Save the figure if foldername is provided
        if(foldername!=None):
            filename = 'section_graph_'+cell_dict['protocol']+'_'+cell_dict['filename'].split('_')[3]+'_'+'cycle'+cycle+'.png'
            fig.savefig(os.path.join('.','figures',foldername,filename), dpi=300, facecolor='white', edgecolor='none')
            
    
def build_all_comparison_graph_v2(X, y, regression, len_list, name, textSize=64, fontName='Microsoft Sans Serif', markerSize=1800, save=100):
    """
    This function generates a comparison graph for model predictions versus true values. It visualizes the training,
    primary test, and secondary test data points, along with a diagonal line representing perfect predictions.
    Additionally, it includes an inset histogram plot showing the residuals (differences between true and predicted values).

    Inputs:
    - X: Feature matrix (numpy array) used for predictions.
    - y: True target values (numpy array).
    - regression: Trained (ElasticNet) regression model used for predictions.
    - len_list: List containing the lengths of training, primary test, and secondary test datasets.
    - name: Name of the graph (used for saving the figure).
    - textSize: Font size for labels and text (default: 64).
    - fontName: Font name for labels and text (default: 'Microsoft Sans Serif').
    - markerSize: Size of scatter plot markers (default: 1800).
    - save: arbitrary value to NOT save the plot (default: 100).

    Outputs:
    - Saves the generated graph to a file with the specified name.
    """
    # Predict values for training, primary test, secondary test, and all data
    y_train_pred = regression.predict(X[0:len_list[0]])
    y_test_pred = regression.predict(X[len_list[0]:(len_list[0]+len_list[1])])
    y_sectest_pred = regression.predict(X[(len_list[0]+len_list[1]):])
    y_all_pred = regression.predict(X)

    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(15,15))

    # Scatter plot for training data
    ax.scatter(10**y[0:len_list[0]], 10**regression.predict(X[0:len_list[0]]), color='forestgreen', label='Train', s=markerSize, marker='o', zorder=1)
    # Scatter plot for primary test data
    ax.scatter(10**y[len_list[0]:(len_list[0]+len_list[1])], 10**regression.predict(X[len_list[0]:(len_list[0]+len_list[1])]), color='steelblue', label='Primary test', s=markerSize, marker='o', zorder=2)
    # Scatter plot for secondary test data
    ax.scatter(10**y[(len_list[0]+len_list[1]):], 10**regression.predict(X[(len_list[0]+len_list[1]):]), color='darkorange', label='Secondary test', s=markerSize, marker='o', zorder=3)
    
    # Diagonal line representing perfect predictions
    ax.plot([0, 2500], [0, 2500], color='black', linestyle='--', linewidth=4, zorder=2)

    # Set axis labels
    ax.set_ylabel("Estimated cycle life", fontsize=textSize+5, fontname=fontName)
    ax.set_xlabel("True cycle life",fontsize=textSize+5, fontname=fontName)

    # Set axis ticks and limits
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels([0, 500, 1000, 1500, 2000], fontname=fontName, y = -0.02)
    ax.set_yticks([500, 1000, 1500, 2000])
    ax.set_yticklabels([500, 1000, 1500, 2000], fontname=fontName, x = -0.02)
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 2500])

    # Add R2-score text to the plot
    ax.text(x=1500, y=300, s="$R^{}$={:.3f}".format('2', r2_score(y_true = 10**y, y_pred = 10**y_all_pred)), 
            fontsize=textSize, zorder=3, fontname=fontName)
    
    # Customize tick parameters
    ax.tick_params(labelsize=textSize, left=False, bottom=False)
    ax.yaxis.set_label_coords(-0.23, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.13)
    plt.grid()

    # Plot for inset residual histogram plot inside the main figure
    inset_ax = inset_axes(ax, width="38%", height="32%",
                          loc="upper left", bbox_to_anchor = (0.09, -0.07, 1, 1), bbox_transform=ax.transAxes)
    inset_ax.set_facecolor('none')
    residual = (10**y - 10**regression.predict(X))
    counts, bins, bars = inset_ax.hist(residual, bins=44, range=(-1100, 1100),
                  color='dimgray', edgecolor='white')

    # Set ticks for the inset histogram
    maxCount = int(counts.max() / 10)
    countTicks = np.arange(0, maxCount*10+1, 10)

    # Set inset-axis ticks and labels
    inset_ax.set_xticks([-1050, -500, 0, 500, 1050])
    inset_ax.set_xticklabels([-1000, -500, 0, 500, 1000], fontname=fontName, y=-0.03)
    inset_ax.set_yticks(countTicks)
    inset_ax.set_yticklabels(countTicks, fontname=fontName, x = -0.03)
    inset_ax.tick_params(labelsize=textSize/11*5+2)
    inset_ax.yaxis.set_tick_params(direction='in', length=15, which='major', width=4)
    inset_ax.xaxis.set_tick_params(direction='out', length=10, which='major', width=4)
    
    # Inset-axis title
    inset_ax.set_title("Residual error (cycle)", fontsize=textSize/11*7, x=0.5, y=1.03)

    # Set inset-axis splines
    for axis in ['top', 'bottom', 'left', 'right']: 
        ax.spines[axis].set_linewidth(4)
        inset_ax.spines[axis].set_linewidth(4)

    # Print R2-scores
    stringOutput = ("\nNormal-scale scores" +
                    "\nAll data score: %.3f" % (r2_score(y_true=10**y, y_pred=10**y_all_pred)) +
                    "\nTrain data score: %.3f" % (r2_score(y_true=10**y[0:len_list[0]], y_pred=10**y_train_pred)) +
                    "\nTest data score: %.3f" % (r2_score(y_true=10**y[len_list[0]:(len_list[0]+len_list[1])], y_pred=10**y_test_pred)) +
                    "\nSecondary test data score: %.3f" % (r2_score(y_true=10**y[(len_list[0]+len_list[1]):], y_pred=10**y_sectest_pred)))
    print(stringOutput)
    
    # Save the figure if save is not 100
    if(save!=100):
        fig.savefig(save, dpi=1000, bbox_inches='tight')
    plt.show()
    return stringOutput

