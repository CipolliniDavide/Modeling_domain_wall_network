'''
This script is used to produce Fig 5A,B,C,D and figure S2 A, B, C.
'''

import os
import sys
sys.path.insert(1, '{:s}/'.format(os.getcwd()))

from sklearn import mixture
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import glob
import json
from neuromorphic_materials.graph_scripts.helpers import utils, graph
from neuromorphic_materials.graph_scripts.helpers.visual_utils import (set_ticks_label, create_colorbar, create_order_of_magnitude_strings, set_legend)
# from neuromorphic_materials.graph_similarity.scripts.helpers_plot import fitness_plot

# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from mpl_toolkits.mplot3d import Axes3D
# import time
# from datetime import datetime
# from pathlib import Path
#
# import networkx as nx

# import pygad
# from tap import Tap
# from varname import nameof
#
# from neuromorphic_materials.graph_similarity.spectral_entropy import (
#     compute_normalized_spectral_entropy,
#     compute_spectral_entropy,
# )
# from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
#     extract_graph_from_binary_voronoi,
# )
# from neuromorphic_materials.sample_generation.src.pdf import uniform_point
# from neuromorphic_materials.sample_generation.src.voronoi.generator import (
#     VoronoiGenerationError,
#     VoronoiGenerator,
# )
# from neuromorphic_materials.graph_similarity.scripts.helpers_graph import (connected_components, merge_nodes_by_distance)


def save_GMM_peak_solutions(solution_data, fitness_data, mean, weights, covs, save_path):
    def select_solutions_in_interval(data, low_bound, up_bound):
        return (data >= low_bound) & (data <= up_bound)

    # Calculate standard deviations (square roots of variances)
    std_deviations = np.sqrt(np.diagonal(covs, axis1=1, axis2=2)).reshape(-1)
    save_GMM_solutions = save_path + '/GMM_solutions/'
    utils.ensure_dir(save_GMM_solutions)
    for ind in range(len(mean)):
        # The index of which gaussian pick to choose is a trade off between higher mean fitness and weight of the gaussian
        # ind = -1

        indexes = select_solutions_in_interval(data=fitness_data.reshape(-1),
                                               low_bound=mean[ind] - std_deviations[ind],
                                               up_bound=mean[ind] + std_deviations[ind])

        sol = solution_data.reshape(-1, solution_data.shape[-1])[indexes]
        print(f'\nNumber of solutions identified by GMM is {np.shape(sol)[0]}')
        best_sol = sol.mean(axis=0)
        best_sol_std = sol.std(axis=0)
        dictionary_of_params = {'d': [], 'p': [], 'beta': []}
        keys = ['d', 'p', 'beta']
        # print('\n')
        for i, ylab in enumerate(ylabel):
            print('{:s}={:.4f} +- {:.4f}'.format(ylab, best_sol[i], best_sol_std[i]))
            dictionary_of_params[keys[i]].append(round(best_sol[i], 4))
            dictionary_of_params[keys[i]].append(round(best_sol_std[i], 4))
        dictionary_of_params['mean_fit'] = fitness_data.reshape(-1)[indexes].mean()
        dictionary_of_params['std_fit'] = fitness_data.reshape(-1)[indexes].std()
        dictionary_of_params['number_of_samples'] = np.shape(sol)[0]
        with open(save_GMM_solutions + f'params_{ind}GMM.json', 'w') as json_file:
            json.dump(dictionary_of_params, json_file)

        print(f"Data has been saved to {save_GMM_solutions + f'params{ind}_GMM.json'}")


def gaussian_mixture_plot(data, number_of_bins=10, name_fig='overFitness', save_fold=None, color='blue', fig_format='.pdf',
                          figsize=(8, 5), labelx='Nodes', labely='Probability Density', show=False, num_of_gaussians=3,
                          x=None, random_seed=42):
    np.random.seed(random_seed)
    if x is None:
        x = np.array(np.linspace(0, data.max(), 320)).reshape(-1, 1)
    else:
        x = np.array(x).reshape(-1, 1)
    temp = np.zeros(((1,) + np.shape(x)))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    data = np.array(data).reshape(-1, 1)
    clf = mixture.GaussianMixture(n_components=num_of_gaussians, covariance_type='full', random_state=random_seed)
    clf.fit(data)
    weights = clf.weights_
    mean = clf.means_
    covs = clf.covariances_

    # sort
    sorted_indexes = np.argsort(mean.reshape(-1))
    mean = mean[sorted_indexes]
    weights = weights[sorted_indexes]
    covs = covs[sorted_indexes]

    ax.hist(data.reshape(-1), density=True, color=color, alpha=.4, bins=number_of_bins)
    col = [f'C{i}' for i in range(num_of_gaussians)]
    # col = ['black', 'black']
    y_ax_sum = np.zeros_like(x)
    for i in range(len(weights)):
        y_axis0 = norm.pdf(x, float(mean[i][0]), np.sqrt(float(covs[i][0][0]))) * weights[i]  # 1st gaussian
        y_ax_sum = y_ax_sum + y_axis0
        ax.axvline(x=mean[i][0], color=col[i], linestyle='--', linewidth=2, alpha=.8)
        ax.plot(x, y_axis0, lw=2, c=col[i], ls='dashed', alpha=.8)
    temp[0] = y_ax_sum
    ax.plot(x, y_ax_sum, lw=2, c=color, alpha=.4)
    set_ticks_label(ax=ax, ax_type='x',
                    data=data,
                    num=10, valfmt="{x:.2f}",
                    ax_label='{:s}'.format(labelx))
    set_ticks_label(ax=ax, ax_type='y',
                    data=y_ax_sum,
                    num=4, valfmt="{x:.2f}",
                    ax_label=r'{:s}'.format(labely))
    # set_legend(ax)

    # Set ticks for the secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis
    # mean_list = np.round(mean.reshape(-1), 2)
    mean_list = np.sort(mean.reshape(-1))
    ax2.set_xticks(mean_list)  # Add a single tick in the middle
    ax2.xaxis.tick_top()  # Position the tick on top
    # Set tick label color to gray
    set_ticks_label(ax=ax2, ax_type='x',
                    data=[0],
                    # add_ticks=[mean],
                    ticks=mean_list,
                    num=4, valfmt="{x:.1f}",
                    ax_label='')

    tick_lab = ['{:.2f}'.format(m) if i % 2 else '{:.2f}\n'.format(m) for i, m in enumerate(mean_list)]
    tick_labels = ax2.get_xticklabels()
    # for xtick, color in zip(ax2.get_xticklabels(), colors):
    for i, (xtick, tex) in enumerate(zip(tick_labels, tick_lab)):
        xtick.set_color(col[i])
        xtick.set_text(tex)
    # Update the tick labels on the plot
    ax2.set_xticklabels(tick_labels)
    plt.tight_layout()
    if save_fold:
        plt.savefig('{:s}/GMM_{:s}.{:s}'.format(save_fold, name_fig, fig_format), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    sorted_indexes = np.argsort(mean.reshape(-1))
    return mean[sorted_indexes], weights[sorted_indexes], covs[sorted_indexes]


def fitness_scatter_plot_across_iterations(y_data, mask, ylabel='', xlabel='', figname='fit_sc_acr_iter',
                                           figformat='.pdf', save_path='', show=False, logy=False, selected_peaks=None,
                                           mean=None, weights=None, covs=None):

    flattened_y_data = y_data[mask]
    batch_indices, x_data, signal_indices = np.where(mask)

    fig, ax = plt.subplots(ncols=1, figsize=(6, 6))
    ax.scatter(x_data, flattened_y_data, alpha=0.15, s=5, c='blue',) # c=batch_indices, cmap='viridis',)

    if selected_peaks is not None:
        # Calculate standard deviations (square roots of variances)
        std_deviations = np.sqrt(np.diagonal(covs, axis1=1, axis2=2)).reshape(-1)
        mean = mean.reshape(-1)
        col = [f'C{i}' for i in range(len(mean))]
        for i in selected_peaks:
            # ax.fill_between(x=np.arange(-3, 0), y1=mean[i]-std_deviations[i], y2=mean[i]+std_deviations[i], color=col[i], alpha=.7)
            ax.fill_between(x=np.arange(y_data.shape[1] + 1, y_data.shape[1] + 4), y1=mean[i] - std_deviations[i],
                            y2=mean[i] + std_deviations[i], color=col[i], alpha=.7)
            ax.fill_between(x=np.arange(-1, y_data.shape[1]+1), y1=mean[i] - std_deviations[i],
                            y2=mean[i] + std_deviations[i],
                            color=col[i], alpha=.08)

    set_ticks_label(ax=ax, ax_type='y', data=y_data,
                    num=6, valfmt="{x:.2f}",
                    ax_label=ylabel,
                    )
    set_ticks_label(ax=ax, ax_type='x', data=np.arange(y_data.shape[1]), ticks=np.arange(y_data.shape[1])[::10],
                    num=5, valfmt="{x:.0f}",
                    ax_label=xlabel,
                    )
    ax.tick_params(axis='both', which='major', width=2, length=7.5)
    if logy:
        plt.yscale('log')
        # plt.xscale('log')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def fitness_scatter_plot_across_iterations_with_histogram(y_data, mask, ylabel='', xlabel='',
                                                          figname='fit_sc_acr_iter',
                                                        figformat='.pdf',
                                                          figsize=(10, 6),
                                                          save_path='',
                                                        color='blue',
                                                        show=False, logy=False, selected_peaks=None,
                                                        labelx_hist='',
                                                        mean=None, weights=None, covs=None,
                                                        labely_hist='Probability Density',
                                                          x=None, number_of_bins=10):

    flattened_y_data = y_data[mask]
    batch_indices, x_data, signal_indices = np.where(mask)

    # fig, [ax, ax1] = plt.subplots(ncols=2, figsize=(12, 6))
    from matplotlib import gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax = plt.subplot(gs[0])

    ax.scatter(x_data, flattened_y_data, alpha=0.15, s=5, c=color,) # c=batch_indices, cmap='viridis',)

    col = [f'C{i}' for i in range(len(mean))][::-1]
    if selected_peaks is not None:
        # Calculate standard deviations (square roots of variances)
        std_deviations = np.sqrt(np.diagonal(covs, axis1=1, axis2=2)).reshape(-1)
        mean = mean.reshape(-1)

        for i in selected_peaks:
            # ax.fill_between(x=np.arange(-3, 0), y1=mean[i]-std_deviations[i], y2=mean[i]+std_deviations[i], color=col[i], alpha=.7)
            ax.fill_between(x=np.arange(y_data.shape[1] + 1, y_data.shape[1] + 4), y1=mean[i] - std_deviations[i],
                            y2=mean[i] + std_deviations[i], color=col[i], alpha=.7)
            ax.fill_between(x=np.arange(-1, y_data.shape[1]+1), y1=mean[i] - std_deviations[i],
                            y2=mean[i] + std_deviations[i],
                            color=col[i], alpha=.08)

    set_ticks_label(ax=ax, ax_type='y', data=y_data,
                    num=6, valfmt="{x:.2f}",
                    ax_label=ylabel,
                    )
    set_ticks_label(ax=ax, ax_type='x', data=np.arange(y_data.shape[1]),
                    ticks=np.arange(y_data.shape[1])[::10],
                    num=5, valfmt="{x:.0f}",
                    ax_label=xlabel,
                    )
    ax.tick_params(axis='both', which='major', width=2, length=7.5)
    if logy:
        plt.yscale('log')
        # plt.xscale('log')

    ax1 = plt.subplot(gs[1])
    set_ticks_label(ax=ax1, ax_type='y', data=y_data,
                    num=6, valfmt="{x:.2f}",
                    ax_label='',
                    )

    temp = np.zeros(((1,) + np.shape(x)))
    ax1.hist(flattened_y_data.reshape(-1), density=True, color=color, alpha=.4, bins=number_of_bins, orientation='horizontal',)
    # col = [f'C{i}' for i in range(args.num_gaussians)]
    # col = ['black', 'black']
    y_ax_sum = np.zeros_like(x)
    for i in range(len(weights)):
        y_axis0 = norm.pdf(x, float(mean[i]), np.sqrt(float(covs[i][0]))) * weights[i]  # 1st gaussian
        y_ax_sum = y_ax_sum + y_axis0
        ax1.axhline(y=mean[i], color=col[i], linestyle='--', linewidth=2, alpha=.8)
        ax1.plot(y_axis0, x, lw=2, c=col[i], ls='dashed', alpha=.8)
    temp[0] = y_ax_sum
    ax1.plot(y_ax_sum, x, lw=2, c=color, alpha=.4)
    set_ticks_label(ax=ax1, ax_type='y',
                    data=fitness_data, only_ticks=True,
                    num=6, valfmt="{x:.2f}", ax_label='',)
                    # ax_label='{:s}'.format(labelx_hist))
    set_ticks_label(ax=ax1, ax_type='x',
                    data=y_ax_sum,
                    num=4, valfmt="{x:.2f}",
                    ax_label=r'{:s}'.format(labely_hist))
    # set_legend(ax)

    # Set ticks for the secondary x-axis
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())  # Match the x-limits of the primary axis
    mean_list = np.sort(mean.reshape(-1))
    ax2.set_yticks(mean_list)  # Add a single tick in the middle
    ax2.yaxis.tick_right()  # Position the tick on top

    tick_lab = ['{:.2f}'.format(m) if i % 2 else '{:.2f}\n'.format(m) for i, m in enumerate(mean_list)]
    tick_labels = ax2.get_yticklabels()
    # for xtick, color in zip(ax2.get_xticklabels(), colors):
    fontdict_ticks_label_standard = {'weight': 'bold', 'size': 'x-large'}
    for i, (xtick, tex) in enumerate(zip(tick_labels, tick_lab)):
        xtick.set_color(col[i])
        xtick.set_text(tex)
        xtick.set_font_properties(fontdict_ticks_label_standard)

    # Update the tick labels on the plot
    ax2.set_yticklabels(tick_labels)

    ax1.tick_params(axis='both', which='major', width=2, length=7.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_scatter(data, color_array, labels, save_path=None, figname='3dscatter', figsize=(16, 9), cbarlab='Fitness',
                 figformat='.pdf', show=False, vmax=2):
    """
    Create a 3D surface plot.

    Args:
        data (array-like): A 2D array where the first three columns represent coordinates.
        color_array (array-like): An array that specifies the color of each point on the surface.
        labels (list): A list of three labels for the x, y, and z axes.

    Returns:
        None
    """
    if data.shape[1] != 3:
        raise ValueError("Input data must have exactly three columns for coordinates.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.3,
            alpha=0.2)

    # Create the 3D surface plot
    sc = ax.scatter(x, y, z, s=25, c=color_array, cmap=plt.get_cmap('hsv'), vmax=vmax,
                    alpha=.6)

    # Add color bar
    # fig, ax, cbar_edg = create_colorbar(fig=fig, ax=ax, mapp=sc, array_of_values=color_array, valfmt="{x:.1f}",
    #                                     fontdict_cbar_label={'label': 'Fitness'})

    # Create a color bar using a separate 2D subplot
    cbar = fig.colorbar(sc, ax=ax, shrink=0.4, aspect=5, pad=.12)
    # cbar.set_label(cbarlab)
    set_ticks_label(ax=cbar.ax, ax_type='y', data=np.clip(color_array, a_min=0, a_max=2),
                    num=4, valfmt="{x:.1f}",
                    ax_label=cbarlab,
                    )
    label_pad=15
    fontdict_ticks_label = {'weight': 'bold', 'size': 'large'}
    # Set labels for the three axes
    set_ticks_label(ax=ax, ax_type='y', data=y,
                    num=5, valfmt="{x:.2f}", fontdict_ticks_label=fontdict_ticks_label,
                    ax_label=labels[1], label_pad=label_pad,
                    )
    set_ticks_label(ax=ax, ax_type='x', data=x, #ticks=x_data[::10],
                    num=4, valfmt="{x:.2f}", label_pad=label_pad+8,
                    ax_label=labels[0], fontdict_ticks_label=fontdict_ticks_label,
                    )
    set_ticks_label(ax=ax, ax_type='z', data=z,  # ticks=x_data[::10],
                    num=4, valfmt="{x:.2f}", label_pad=label_pad,
                    ax_label=labels[2], fontdict_ticks_label=fontdict_ticks_label,
                    )

    # ax.set_xlabel(labels[0])
    # ax.set_ylabel(labels[1])
    # ax.set_zlabel(labels[2])

    ax.view_init(-160, 15)

    # fig.tight_layout()
    # plt.show()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    a=0

def compute_mean_with_nan(selected_values):
    # Get the shape of the selected_values array
    shape = selected_values.shape

    # Initialize the result array with NaN values
    y_data = np.empty((shape[0], shape[1]))

    for i in range(shape[0]):
        for j in range(shape[1]):
            # Extract the slice along the third axis
            slice_ = selected_values[i, j, :]

            # Check if there are NaN values in the slice
            # try:
            if np.isnan(slice_).sum() == len(slice_):
                y_data[i, j] = np.nan
            else:
                y_data[i, j] = np.nanmean(slice_)
            # except:
            #   a=0

    return y_data

def trajectories_plot(y_data, x_data, save_path=None, figname='fitness', figsize=(6, 4), ylabel='Fitness',
                 xlabel='Generation', log=None, color='blue', alpha=0.08,
                 figformat='.svg', show=False, max_fitness=True):
    print(f'\n{ylabel}')
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    for i, y in enumerate(y_data):
        print(f'Traj {i}: convergence to {y[-3:]}')
        ax.plot(x_data, y, c=color, alpha=alpha, linewidth=3)
    # plt.margins(x=0)
    # ax.plot(x_data, y_data, c=color, linewidth=4)
    # if yerr is None:
    #     pass
    # else:
    #     ax.fill_between(x_data, y_data - yerr, y_data + yerr, alpha=0.2, color=color,
    #                                      # label='Standard Deviation')
    #                     )
    #     ydata_ticks = [np.min(y_data - yerr), np.max(y_data + yerr)]
    # # ax.set_xlabel('Generation')
    # ax.set_ylabel('Fitness')
    # # Set font properties for the y-axis label
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    # y_label = ax.yaxis.get_label()
    # y_label.set_font_properties(font_properties)
    # x_label = ax.xaxis.get_label()
    # x_label.set_font_properties(font_properties)
    set_ticks_label(ax=ax, ax_type='y', data=y_data,
                    num=5, valfmt="{x:.2f}",
                    ax_label=ylabel,
                    )
    set_ticks_label(ax=ax, ax_type='x', data=x_data, ticks=x_data[::10],
                    num=4, valfmt="{x:.0f}",
                    ax_label=xlabel,
                    )

    ax.tick_params(axis='both', which='major', width=2, length=7.5)

    if log:
        plt.yscale('log')
        plt.xscale('log')

    # ax.tick_params(axis='both', which='minor', width=2, length=4)
    # set_legend(ax, title='', ncol=1, loc=0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    a=0

def fitness_plot(y_data, x_data, yerr=None, save_path=None, figname='fitness', figsize=(6, 4), ylabel='Fitness',
                 xlabel='Generation', log=None, color='blue',
                 figformat='.svg', show=False, max_fitness=True):

    # if max_fitness:
    #     y_data = np.array(ga.saved_fitnesses).max(axis=1)
    # else:
    #     y_data = np.array(ga.saved_fitnesses).mean(axis=1)
    # x_data = range(ga.generations_completed)

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    ax.plot(x_data, y_data, c=color, linewidth=4)
    if yerr is None:
        pass
    else:
        ax.fill_between(x_data, y_data - yerr, y_data + yerr, alpha=0.2, color=color,
                                         # label='Standard Deviation')
                        )
        ydata_ticks = [np.min(y_data - yerr), np.max(y_data + yerr)]
    # ax.set_xlabel('Generation')
    # ax.set_ylabel('Fitness')
    # # Set font properties for the y-axis label
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    # y_label = ax.yaxis.get_label()
    # y_label.set_font_properties(font_properties)
    # x_label = ax.xaxis.get_label()
    # x_label.set_font_properties(font_properties)
    set_ticks_label(ax=ax, ax_type='y', data=ydata_ticks,
                    num=6, valfmt="{x:.2f}",
                    ax_label=ylabel,
                    )
    set_ticks_label(ax=ax, ax_type='x', data=x_data, ticks=x_data[::10],
                    num=5, valfmt="{x:.0f}",
                    ax_label=xlabel,
                    )

    ax.tick_params(axis='both', which='major', width=2, length=7.5)

    if log:
        plt.yscale('log')
        plt.xscale('log')

    # ax.tick_params(axis='both', which='minor', width=2, length=4)
    # set_legend(ax, title='', ncol=1, loc=0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def create_combined_plot(inner_y, x_data, y_data, save_path=None, figname='combined_plot', figsize=(10, 6),
                         yerr=None,
                         ylabel='Fitness', xlabel='Generation', log=None, color='blue', alpha=0.08,
                         figformat='.svg', show=False, max_fitness=True,
                         inset_pos=[0.6, 0.6, 0.35, 0.35]):


    def fitness_plot_inner(ax, x_data, y_data, yerr=None, color='blue'):
        for y in y_data:
            ax.plot(x_data, y, c=color, linewidth=2, alpha=alpha)

    fig, ax1 = plt.subplots(ncols=1, figsize=figsize)
    ax1.plot(x_data, y_data, c=color, alpha=1, linewidth=3)
    if yerr is not None:
        ax1.fill_between(x_data, y_data - yerr, y_data + yerr, alpha=0.2, color=color)

    set_ticks_label(ax=ax1, ax_type='y', data=y_data, num=5, valfmt="{x:.2f}", ax_label=ylabel)
    set_ticks_label(ax=ax1, ax_type='x', data=x_data, ticks=x_data[::10], num=4, valfmt="{x:.0f}", ax_label=xlabel)

    ax1.tick_params(axis='both', which='major', width=2, length=7.5)

    if log:
        plt.yscale('log')
        plt.xscale('log')

    ax1.set_ylabel(ylabel, weight='bold', size='xx-large')

    # Create an inset for fitness plot
    inset = fig.add_axes(inset_pos)  # Define the inset position and size
    fitness_plot_inner(inset, x_data, inner_y, color=color)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig('{:s}{:s}{:s}'.format(save_path, figname, figformat), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', '--load_folder_path',
                        default='{:s}/PycharmProjects/Modeling_domain_wall_network/'.format(os.path.expanduser('~'))+'data/iter20_eps1.4/', type=str)
    parser.add_argument('-svp', '--save_path',
                        default='{:s}/PycharmProjects/Modeling_domain_wall_network/'.format(os.path.expanduser('~'))+'Figures/optimization/iter20_eps1.4/', type=str)
    parser.add_argument('-p_th', '--p_threshold', default=.7, type=float,
                        help='To set a threshold to select either the horizontal solutions')
    parser.add_argument('-n_gauss', '--num_gaussians', default=6, type=int, help='Number of gaussians to detect peaks.')
    parser.add_argument('-figform', '--fig_format', default='.jpg', type=str)
    parser.add_argument('-show', '--show_fig', default=False, type=bool)
    args = parser.parse_args()

    save_path = args.save_path
    utils.ensure_dir(save_path)
    print(f'Save folder: {save_path}')

    load_folder_path = args.load_folder_path
    print(f'Load folder: {args.load_folder_path}')
    # Use glob to recursively find files with .pkl extension
    file_list = sorted(glob.glob(f'{load_folder_path}/**/*/*.pkl', recursive=True))

    fitness_data = []
    solution_data = []

    # Exclude element from file_list
    # index_to_exclude = 3
    # file_list = file_list[:index_to_exclude] + file_list[index_to_exclude + 1:]

    for file_name in file_list:
        ga = pickle.load(open(file_name, "rb"))
        fitness_data.append(ga.saved_fitnesses)
        solution_data.append(ga.saved_solutions)
        # print(np.shape(ga.saved_fitnesses))
        # fitness_plot(ga, save_path=load_folder, figname=file_name.strip('.pkl')+'maxfitness', figformat='.svg', figsize=(6, 4),
        #              show=True, max_fitness=True)
        # fitness_plot(ga, save_path=load_folder, figname=file_name.strip('.pkl')+'meanfitness', figformat='.svg', figsize=(6, 4),
        #              show=True, max_fitness=False)
    fitness_data = np.array(fitness_data)
    solution_data = np.array(solution_data)


    ######################### Plot fitness evolution across optimization ###############################################
    # Fig 5A
    ydata_mean_overNbest = fitness_data.mean(axis=2)
    fitness_plot(x_data=range(fitness_data.shape[1]), y_data=ydata_mean_overNbest.mean(axis=0),
                 yerr=ydata_mean_overNbest.std(axis=0), ylabel='N-best fitness',
                 save_path=save_path, figname='Nbestfitness', figformat=args.fig_format, show=args.show_fig,
                 figsize=(6, 4),)
    # Fig 5B
    ydata_max = fitness_data.max(axis=2)
    fitness_plot(x_data=range(fitness_data.shape[1]),
                 y_data=ydata_max.mean(axis=0),
                 yerr=ydata_max.std(axis=0), ylabel='Max Fitness',
                 save_path=save_path, figname='maxfitness', figformat=args.fig_format, show=args.show_fig,
                 figsize=(6, 4), )


    #################################### Parameters evolution across optimization ######################################
    # Fig 5C
    ylabel = ['Site density\nd', 'Horiz. prob\np', 'Cheb. scal.\n'+r'$\mathbf{\beta}$']
    fig_name = ['node_dens', 'HorizProb', 'beta']
    set_log = [None, None, None]
    color = ['indigo', 'darkgoldenrod', 'green']
    inset_pos = [[.6, .6, .35, .35],
                 [.55, .4, .4, .4],
                 [.55, .3, .4, .4]]

    plot_3d_scatter(data=solution_data.reshape(-1, solution_data.shape[-1]),
                    color_array=fitness_data.reshape(-1), labels=ylabel,
                    save_path=save_path, figformat='.svg', show=args.show_fig, figsize=(8, 8)
                    )

    ######################### Plot trajectories ##########################################
    # For each simulation optimization we average over the N=10 best solution (axis=2) and we plot all trajectories,
    # one for each optimization run.

    for i, ylab in enumerate(ylabel):
        y_data = solution_data[..., i].mean(axis=2).reshape((solution_data.shape[0], solution_data.shape[1]))
        # y_data = solution_data[..., i].reshape((solution_data.shape[0]*solution_data.shape[2], solution_data.shape[1]))
        trajectories_plot(x_data=range(fitness_data.shape[1]),
                          y_data=y_data,
                          ylabel=ylab, color=color[i], alpha=0.4,
                          save_path=save_path, figname='trajecory' + fig_name[i],
                          figformat=args.fig_format,
                          show=args.show_fig,
                          figsize=(5, 4), )

    ############### Plot mean evolution ##########################
    # Fig S2 is produced with "create_combined_plot".

    # Here we average over the optimization runs.
    # Nevertheless, we restrict the plot to optimization runs that converged towards p>p_threshold.
    # In fact, being our method rotation invariant, the configuration with p is identical from the point of view of the
    # fitness score to the configuration 1-p.

    threshold_p = args.p_threshold
    # Get the last 5 values of convergence
    last_5_values = solution_data[..., 1].mean(axis=2)[:, -5:]
    if args.p_threshold > .5:
        mask_over_runs = np.all(last_5_values < threshold_p, axis=1)
    else:
        mask_over_runs = np.all(last_5_values > threshold_p, axis=1)
    # Find the indices where the condition is met
    indices = np.where(mask_over_runs)[0]

    for i, ylab in enumerate(ylabel):
        y_data = solution_data[indices, ..., i].mean(axis=2)
        fitness_plot(x_data=range(fitness_data.shape[1]), color=color[i],
                     y_data=y_data.mean(axis=0),
                     yerr=y_data.std(axis=0) / np.sqrt(y_data.shape[0]), ylabel=ylab,
                     save_path=save_path, figname=f'avg_traj_removed_opt_pminor{threshold_p}' + fig_name[i],
                     figformat=args.fig_format,
                      show=args.show_fig,
                      figsize=(5, 4), )

        # Fig S2
        create_combined_plot(x_data=range(fitness_data.shape[1]),
                             inner_y=solution_data[..., i].mean(axis=2),
                             color=color[i],
                             y_data=y_data.mean(axis=0),
                             yerr=y_data.std(axis=0) / np.sqrt(y_data.shape[0]),
                             ylabel=ylab, alpha=.4,
                             save_path=save_path,
                             figname=f'inset_avg_traj_plot_pminr{threshold_p}' + fig_name[i],
                             figformat=args.fig_format,
                             show=args.show_fig,
                             figsize=(5, 4), inset_pos=inset_pos[i]
                            )

    #####################################################################################
    # Select solution, over all runs and generation (!), that are either vertical or horizontal.
    # Thus, we create a second masks over values which results in an inhomogeneous array.
    # Also, the means and the std are then computed over a different number of samples.
    # To this purpose we created the function compute_mean_with_nan.

    if args.p_threshold > .5:
        mask = solution_data[..., 1] < args.p_threshold
    else:
        mask = solution_data[..., 1] > args.p_threshold

    for i, ylab in enumerate(ylabel):
        selected_values = np.full_like(solution_data[..., i], np.nan, dtype=float)
        # Assign values from data that meet the condition to selected_values
        selected_values[mask] = solution_data[..., i][mask]
        y_data = compute_mean_with_nan(selected_values)
        fitness_plot(x_data=range(fitness_data.shape[1]),
                     y_data=np.nanmean(y_data, axis=0),
                     yerr=np.nanstd(y_data, axis=0)/np.sqrt(y_data.shape[0]-np.sum(np.isnan(y_data), axis=0)),
                     ylabel=ylab, color=color[i],
                     save_path=save_path, figname=fig_name[i], figformat=args.fig_format, show=args.show_fig,
                     log=set_log[i],
                     figsize=(5, 4), )

    ################################## Gaussian Mixture selection method ################################################################
    # We use GMM to select from the fitness scores over all runs and over all generation. We cluster the data in
    # multiple gaussians. The selected gaussian is a trade-off between its mean value and its weight. We then perform
    # the mean of the solutions corresponding to fitness values that are in one standard deviation from the gaussian
    # peak.
    # Before using GMM we already selected only the solution tuples where p>=.5 as our solution are
    # rotationally invariant and the majority of our runs end up in p~.8. The remaining part of the solutions
    # gave us 1-p, and results in equivalent fitness scores.
    # The selection is carried by the boolean array mask.

    if args.p_threshold > .5:
        mask = solution_data[..., 1] <= .5
    else:
        mask = solution_data[..., 1] >= .5

    mean, weights, covs = gaussian_mixture_plot(data=fitness_data[mask], number_of_bins=100, name_fig='overFitness',
                                                color='blue', fig_format='.pdf', figsize=(10, 5), labelx='Fitness',
                                                labely='Probability Density', save_fold=save_path,
                                                num_of_gaussians=args.num_gaussians, random_seed=3,
                                                x=np.linspace(0, 2.5, 320),
                                                # show=True,)
                                                show=args.show_fig, )

    # fitness_scatter_plot_across_iterations(y_data=fitness_data, mask=mask, xlabel='Generation', ylabel='N-best fitness',
    #                                        mean=mean, weights=weights, covs=covs,
    #                                        logy=False, figname='fit_sc_acr_iter',
    #                                        selected_peaks=np.arange(len(mean)-1),
    #                                        figformat=args.fig_format,
    #                                        save_path=save_path,
    #                                        # show=True)
    #                                        show=args.show_fig, )

    # Fig 5D
    fitness_scatter_plot_across_iterations_with_histogram(y_data=fitness_data, mask=mask, xlabel='Generation',
                                                          ylabel='N-best fitness',
                                                          mean=mean, weights=weights, covs=covs,
                                                          logy=False, figname='fit_and_hist_sc_acr_iter',
                                                          # args.num_gaussians=args.num_gaussians,
                                                          # random_seed=3,
                                                          x=np.linspace(0, 2.5, 320),
                                                          selected_peaks=np.arange(len(mean) - 1),
                                                          figformat=args.fig_format,
                                                          save_path=save_path,
                                                          labely_hist='Probability Density',
                                                          number_of_bins=100,
                                                          figsize=(10, 6),
                                                          # show=True)
                                                           show=args.show_fig, )

    # Save solutions found through GMM selection
    save_GMM_peak_solutions(solution_data=solution_data[mask], fitness_data=fitness_data[mask], save_path=save_path,
                            mean=mean, weights=weights, covs=covs)

    ####################################################################################################################
    ###### The call to save_GMM_peak_solution conlcludes the methods for the paper! ####################################
    ####################################################################################################################

    # ## Other stuff ... Not included in the paper
    # ####################################################################################################################
    # ################################## Best of the best #up_bound=mean[ind] + covs[ind]###############################################################
    # # We select the N-best of the best fitness scores over all runs and over all generation, and we plot the averge solution.
    # # To the purpose of selection, we consider only the solutions that correspond to p>p_thesh as our solution are rotationally
    # # invariant and the majority of our runs end up in p~.8. The remaining part of the solutions give us 1-p,
    # # and results in equivalent fitness scores.
    # # The selection is carried by the boolean array mask.
    #
    # N = 20*fitness_data.shape[0]*fitness_data.shape[-1]
    # top_n_idx = np.argpartition(fitness_data[mask].reshape(-1), -N)[-N:]
    # # N best fitness values (not ordered)
    # best_of_all_fit = fitness_data[mask].reshape(-1)[top_n_idx]
    # mean_fit = np.mean(best_of_all_fit)
    # print(f'\n\nMean beast {N} fit: {mean_fit}')
    #
    # c, bins, _ = plt.hist(fitness_data[mask].reshape(-1), color='blue', bins=100)
    # # plt.hist(bins, best_of_all_fit, color='red')
    # plt.tight_layout()
    # plt.savefig(save_path+f'{N}best_fit.pdf', dpi=300)
    #
    # dictionary_of_params = {'d': [], 'p': [], 'beta': []}
    # keys = ['d', 'p', 'beta']
    # print('\n')
    # for i, ylab in enumerate(ylabel):
    #     best_par = solution_data[mask][..., i].reshape(-1)[top_n_idx]
    #
    #     print('{:s}={:.4f} +- {:.4f}\n'.format(ylab, np.mean(best_par), np.std(best_par)))
    #     dictionary_of_params[keys[i]].append(round(np.mean(best_par), 4))
    #     dictionary_of_params[keys[i]].append(round(np.std(best_par), 4))
    #     # plt.hist(best_par, bins=10)
    #     # plt.ylabel(ylab)
    #     # plt.show()
    #
    # with open(save_path+'Nbest_params.json', 'w') as json_file:
    #     json.dump(dictionary_of_params, json_file)
    #
    # print(f"Data has been saved to {save_path+'Nbest_params.json'}")