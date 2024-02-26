#! /usr/bin/env python3
import os
import sys
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two directories up
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import levy
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from scipy.stats import gaussian_kde

from neuromorphic_materials.graph_scripts.helpers import visualize
from neuromorphic_materials.graph_scripts.helpers import utils, graph
from neuromorphic_materials.graph_scripts.helpers.visual_utils import set_ticks_label, create_order_of_magnitude_strings, set_legend
from neuromorphic_materials.graph_scripts.helpers.kmeans_tsne import plot_clustered_points, plot_k_means_tsn_points
from helpers.graph import (merge_nodes_by_distance, connected_components)

def plot_overlapped_SE_MeanSTD(specEnt_list_gt, specEnt_list_vor, beta_values, curve_labels=['BFO', 'Voronoi'],
                               figsize=(6,4), save_folder=None, fig_name=None, show=False, ylabel="S"):
    # log_min = -3
    # log_max = 4
    # ticks = np.logspace(start=log_min, stop=log_max, num=4, endpoint=True)
    # ticks = [10e-3, .1, 10e2, 10e4]
    # ticks_lab = [r'$\mathbf{10^{-3}}$', r'$\mathbf{10^{-1}}$', r'$\mathbf{10^{1}}$', r'$\mathbf{10^{4}}$']
    # One plot Spectral Entropy
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    for i, se in enumerate([specEnt_list_gt, specEnt_list_vor]):
        # Calculate mean and standard deviation across curves
        mean_curve = np.mean(se, axis=0)
        std_curve = np.std(se, axis=0)
        # Create the plot
        ax.semilogx(beta_values, mean_curve, linewidth=4, label=curve_labels[i], color=colors[i])
        ax.fill_between(beta_values, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=colors[i])
    # ax2.grid()
    # for ax in [ax1, ax2]:
    #     set_ticks_label(ax=ax, ax_type='x', data=[log_min, log_max],
    #                     num=4, valfmt="{x:s}", ticks=ticks,
    #                     only_ticks=False, tick_lab=ticks_lab,
    #                     ax_label=r'$\mathbf{\tau}$', scale='log')
    #
    set_ticks_label(ax=ax, ax_type='y', data=[np.min(specEnt_list_gt), np.min(specEnt_list_vor),
                                              np.max(specEnt_list_gt), np.max(specEnt_list_vor)],
                    num=4, valfmt="{x:.1f}",
                    ax_label=ylabel,
                    )
    ax.set_xscale('log')
    # y_tick_labels = ax.get_yticklabels()
    # for label in y_tick_labels:
    #     label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})

    x_tick_labels = ax.get_xticklabels()
    for label in x_tick_labels:
        label.set_font_properties({'weight': 'bold', 'size': 'large'})
    ax.set_xlabel(r'$\mathbf{\tau}$')

    # Set font properties for the y-axis label
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    y_label = ax.yaxis.get_label()
    y_label.set_font_properties(font_properties)
    x_label = ax.xaxis.get_label()
    x_label.set_font_properties(font_properties)
    ax.tick_params(axis='both', which='both', width=1.7, length=7.5)
    ax.tick_params(axis='both', which='minor', width=1.7, length=4)
    set_legend(ax, title='', ncol=1, loc=0)

    plt.tight_layout()
    if save_folder:
        plt.savefig(save_folder + fig_name)
    if show:
        plt.show()
    else:
        plt.close()
    a=0


def plot__eigenPdf(graph_list_gt, graph_list_vor, save_path=None, curve_labels=['BFO', 'Voronoi'],figname=None, figsize=(6,4), show=False):
    eigen_list_vor = list()
    eigen_list_gt = list()
    for gt in graph_list_gt:
        eigen_list_gt.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))
    for vor in graph_list_vor:
        eigen_list_vor.append(eigh(nx.laplacian_matrix(vor).toarray(), eigvals_only=True))

    flattened_list_gt = np.clip([item for sublist in eigen_list_gt for item in sublist], 0, 1e15)
    flattened_list_vor = np.clip([item for sublist in eigen_list_vor for item in sublist], 0, 1e15)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    # Compute PDF using kernel density estimation (KDE)
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_gt, bins=80)
    ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[0], color=colors[0])
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_vor, bins=bins)
    ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[1], color=colors[1])

    # kde = gaussian_kde(flattened_list)
    # bins = np.linspace(np.min(flattened_list), np.max(flattened_list), 100)
    # pdf = kde(bins)
    # ax.plot(bins, pdf, alpha=0.9, marker='o', linewidth=3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    y_tick_labels = ax.get_yticklabels()
    for label in y_tick_labels:
        label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
    x_tick_labels = ax.get_xticklabels()
    for label in x_tick_labels:
        label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})

    ax.set_xlabel(r'$\mathbf{\lambda}$')
    ax.set_ylabel(r'$\mathbf{P(\lambda})$')
    # Set font properties for the y-axis label
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    y_label = ax.yaxis.get_label()
    y_label.set_font_properties(font_properties)
    x_label = ax.xaxis.get_label()
    x_label.set_font_properties(font_properties)
    ax.tick_params(axis='both', which='both', width=2, length=7.5)
    ax.tick_params(axis='both', which='minor', width=2, length=4)
    set_legend(ax, title='', ncol=1, loc=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + 'All'+figname)
    if show:
        plt.show()
    else:
        plt.close
    a=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp_GT', '--load_path_GT', default='../Dataset/GroundTruth2/graphml', type=str)
    parser.add_argument('-lp_VOR', '--load_path_Vor',
                        default='../Dataset/Voronoi128_d0.4156_p0.8828_beta2.9977/graphml', type=str)
    parser.add_argument('-svp', '--save_path', default=None, type=str)
    parser.add_argument('-eps', '--merge_epsilon', default=2, type=float)
    parser.add_argument('-figform', '--fig_format', default='svg', type=str)
    parser.add_argument('-show', '--show_fig', default=False, type=bool)
    args = parser.parse_args()

    fig_format = args.fig_format
    merge_epsilon = args.merge_epsilon
    labels = ['BFO', 'Voronoi']
    colors = ['blue', 'red']

    if args.save_path is None:
        root = os.getcwd().rsplit('/', 2)[0] + '/'
        save_folder = root + 'StatAnalysis/GT_Vor{:s}/eps{:.1f}'.format(args.load_path_Vor.split('Voronoi')[1].split('/')[0],
                                                                   merge_epsilon)
    else:
        save_folder = '{:s}/eps{:.1f}'.format(args.save_path, merge_epsilon)

    file_list_gt = sorted(glob('{:s}/*.graphml'.format(args.load_path_GT)))
    file_list_vor = sorted(glob('{:s}/*.graphml'.format(args.load_path_Vor)))  # [:300]

    # Load graphs
    graph_list_gt = list()
    graph_list_vor = list()
    crop_name = list()

    for i, fgt in enumerate(file_list_gt):
        crop_name.append(fgt.rsplit('/')[-1].split('.')[0])
        graph_list_gt.append(connected_components(nx.read_graphml(fgt))[0])
    for i, fvor in enumerate(file_list_vor):
        graph_list_vor.append(connected_components(merge_nodes_by_distance(nx.read_graphml(fvor), epsilon=merge_epsilon, return_removed_nodes=False))[0])

    # Create lists
    spectral_ent_gt = list()
    spectral_ent_vor = list()
    log_min = -3  # Start value (10^-3)
    log_max = 4  # Stop value (10^4)
    num_points = 300  # Number of points

    # Generate the log-spaced array
    log_spaced_array = np.logspace(log_min, log_max, num=num_points)
    for gt in graph_list_gt:
        spectral_ent_gt.append(graph.entropy(L=nx.laplacian_matrix(gt).toarray(), beta_range=log_spaced_array))
    for vor in graph_list_vor:
        spectral_ent_vor.append(graph.entropy(L=nx.laplacian_matrix(vor).toarray(), beta_range=log_spaced_array))
    spectral_ent_gt = np.clip(spectral_ent_gt, 0, 1e16)
    spectral_ent_vor = np.clip(spectral_ent_vor, 0, 1e16)

    # Spectral entropy
    spectral_ent_fold = save_folder + '/SE/'
    utils.ensure_dir(spectral_ent_fold)
    plot_overlapped_SE_MeanSTD(specEnt_list_gt=spectral_ent_gt, specEnt_list_vor=spectral_ent_vor,
                               # ylabel=r'$\mathbf{S(\tau)}$',
                               ylabel=r'$\mathbf{S}$',
                               beta_values=log_spaced_array,
                               save_folder=spectral_ent_fold, fig_name='overlapped_SE.{:s}'.format(fig_format),
                               figsize=(6, 4), curve_labels=labels, show=args.show_fig)
    plot__eigenPdf(graph_list_gt=graph_list_gt, graph_list_vor=graph_list_vor, save_path=spectral_ent_fold,
                   figname='eigenPdf.{:s}'.format(fig_format), curve_labels=labels, figsize=(6, 4), show=args.show_fig)
    print(f'Figures spectral analysis saved to:\n\t{spectral_ent_fold}')
    a=0

