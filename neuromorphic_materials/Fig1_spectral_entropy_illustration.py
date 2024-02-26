#! /usr/bin/env python3
"""
This script produces Figure 1 in the paper.
Figures will be saved in SE_illustration/
"""

import os
import sys
# sys.path.insert(1, os.path.abspath(os.getcwd().rsplit('/', 1)[0]))

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
from scipy.linalg import expm
from scipy.stats import gaussian_kde

from graph_scripts.helpers import visualize
from graph_scripts.helpers import utils, graph
from graph_scripts.helpers.visual_utils import set_ticks_label, create_order_of_magnitude_strings, set_legend, create_colorbar
from graph_scripts.helpers.kmeans_tsne import plot_clustered_points, plot_k_means_tsn_points
from graph_stat_analysis.helpers.graph import (merge_nodes_by_distance, connected_components)


def specific_heat(spec_ent, tau_range):
    # Calculate the first derivative of spec_entropy with respect to time
    return - np.diff(spec_ent) / np.diff(np.log(tau_range))

def generate_erdos_renyi_graph(n, p):
    """
    Generate an Erdős-Rényi random graph.

    Parameters:
    - n: Number of nodes
    - p: Probability of edge creation between nodes

    Returns:
    - G: Erdős-Rényi graph
    """
    G = nx.erdos_renyi_graph(n, p)
    return G

def plot_overlapped_SE_MeanSTD(specEnt_list, tau_range,
                               curve_labels=['BFO', 'Grid', 'Erdős–Rényi'],
                               figsize=(6,4), save_folder=None, fig_name=None, show=False, ylabel="S"):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, sharex=True, sharey=True)
    for i, se in enumerate(specEnt_list):
        ax.semilogx(tau_range, se, linewidth=3, label=curve_labels[i], color=colors[i])

    ax.grid()
    set_ticks_label(ax=ax, ax_type='y', data=[0, np.max(se)],
                    num=4, valfmt="{x:.1f}",
                    ax_label=ylabel,
                    )
    ax.set_xscale('log')

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
    # set_legend(ax, title='', ncol=1, loc=0)

    plt.tight_layout()
    if save_folder:
        plt.savefig(save_folder + fig_name)
    if show:
        plt.show()
    else:
        plt.close()
    a=0


# def plot__eigenPdf(graph_list_grid, graph_list_gt, graph_list_random, save_path=None,
#                    curve_labels=['Grid', 'BFO', 'Erdős–Rényi'],
#                    figname=None, figsize=(6,4), show=False):
#
#     eigen_list_vor = list()
#     eigen_list_gt = list()
#     for gt in graph_list_grid:
#         eigen_list_gt.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))
#     for vor in graph_list_gt:
#         eigen_list_vor.append(eigh(nx.laplacian_matrix(vor).toarray(), eigvals_only=True))
#     eigen_list_random = list()
#     for gt in graph_list_random:
#         eigen_list_random.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))
#
#     flattened_list_rand = np.clip([item for sublist in eigen_list_random for item in sublist], 0, 1e15)
#     flattened_list_gt = np.clip([item for sublist in eigen_list_gt for item in sublist], 0, 1e15)
#     flattened_list_vor = np.clip([item for sublist in eigen_list_vor for item in sublist], 0, 1e15)
#
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
#     # Compute PDF using kernel density estimation (KDE)
#     mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_gt, bins=80)
#     ind = mypdf > 0
#     ind[-3:] = True
#     ax.plot(bins[1:][ind], mypdf[ind], alpha=1, linewidth=4, label=curve_labels[0], color=colors[0])
#     # ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[0], color=colors[0])
#     # plt.show()
#     mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_vor, bins=bins)
#     ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[1], color=colors[1])
#     mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_rand, bins=bins)
#     ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[2], color=colors[2])
#
#     # kde = gaussian_kde(flattened_list)
#     # bins = np.linspace(np.min(flattened_list), np.max(flattened_list), 100)
#     # pdf = kde(bins)
#     # ax.plot(bins, pdf, alpha=0.9, marker='o', linewidth=3)
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     y_tick_labels = ax.get_yticklabels()
#     for label in y_tick_labels:
#         label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
#     x_tick_labels = ax.get_xticklabels()
#     for label in x_tick_labels:
#         label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
#
#     ax.set_xlabel(r'$\mathbf{\lambda}$')
#     ax.set_ylabel(r'$\mathbf{P(\lambda})$')
#     # Set font properties for the y-axis label
#     font_properties = {'weight': 'bold', 'size': 'xx-large'}
#     y_label = ax.yaxis.get_label()
#     y_label.set_font_properties(font_properties)
#     x_label = ax.xaxis.get_label()
#     x_label.set_font_properties(font_properties)
#     ax.tick_params(axis='both', which='both', width=2, length=7.5)
#     ax.tick_params(axis='both', which='minor', width=2, length=4)
#     set_legend(ax, title='', ncol=1, loc=0)
#     plt.tight_layout()
#     plt.show()
#     if save_path is not None:
#         plt.savefig(save_path + 'All'+figname)
#     if show:
#         plt.show()
#     else:
#         plt.close
#     a=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default=None, type=str)
    parser.add_argument('-figform', '--fig_format', default='svg', type=str)
    parser.add_argument('-show', '--show_fig', default=True, type=bool)
    args = parser.parse_args()

    fig_format = args.fig_format
    labels = ['Dense Erdős–Rényi', 'Sparse Erdős–Rényi', 'Stochastic-block-model']
    colors = ['blue', 'purple', 'green']

    if args.save_path is None:
        root = os.getcwd().rsplit('/', 2)[0] + '/'
        save_folder = root + 'SE_illustration/'
    else:
        save_folder = '{:s}'.format(args.save_path)
    utils.ensure_dir(save_folder)

    ################# Draw networks in figure 1B ##################
    rand_graph_dense = nx.erdos_renyi_graph(n=8*3, p=.8)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx(G=rand_graph_dense, pos=nx.spring_layout(rand_graph_dense), with_labels=False, node_size=300,
                     node_color=colors[1], ax=ax)
    plt.tight_layout()
    ax.axis('off')
    plt.savefig(save_folder + 'rand_graph.{:s}'.format(fig_format), bbox_inches='tight', dpi=300)
    plt.show()

    sizes = [8, 8, 8]
    inter_p = 0.05
    probs = [[1, inter_p, inter_p], [inter_p, 1, inter_p], [inter_p, inter_p, 1],
             ]
    sbm = nx.stochastic_block_model(sizes, probs, seed=0)
    # sbm.add_nodes_from(list(n + 1 for n in range(100+3)))
    fig, ax = plt.subplots(figsize=(6,5))
    nx.draw_networkx(G=sbm, pos=nx.spring_layout(sbm), with_labels=False, node_size=300, node_color=colors[0], ax=ax)
    plt.tight_layout()
    ax.axis('off')
    plt.savefig(save_folder+'SBM.{:s}'.format(fig_format), bbox_inches='tight', dpi=300)
    plt.show()


    ############################################  Spectral entropy  ########################################################################

    # Generate the log-spaced array
    log_min = -3  # Start value (10^-3)
    log_max = 3  # Stop value (10^4)
    num_points = 300  # Number of points
    tau_range = np.logspace(log_min, log_max, num=num_points)

    # sizes = [100, 100, 200, 100, 200]
    # inter_p = .001 #.01
    # intra_p = .95
    # probs = [[intra_p, inter_p, inter_p, inter_p, inter_p],
    #          [inter_p, intra_p, inter_p, inter_p, inter_p],
    #          [inter_p, inter_p, intra_p, inter_p, inter_p],
    #          [inter_p, inter_p, inter_p, intra_p, inter_p],
    #          [inter_p, inter_p, inter_p, inter_p, intra_p]]
    sizes = [25, 25, 40]
    inter_p = .001  # .01
    intra_p = .6
    probs = [[intra_p, inter_p, inter_p], [inter_p, intra_p, inter_p], [inter_p, inter_p, intra_p]]

    sbm = nx.stochastic_block_model(sizes, probs, seed=0)
    # fig, ax = plt.subplots(figsize=(6,5))
    # nx.draw_networkx(G=sbm, pos=nx.spring_layout(sbm), with_labels=False, node_size=5, node_color=colors[0], ax=ax)
    # plt.tight_layout()
    # ax.axis('off')
    # plt.show()

    rand_graph_sparse = nx.erdos_renyi_graph(n=np.sum(sizes), p=.8)
    spectral_ent_fold = save_folder #+ '/SE_RandN_GRID/'
    utils.ensure_dir(spectral_ent_fold)
    plot_overlapped_SE_MeanSTD(specEnt_list=[graph.entropy(L=nx.laplacian_matrix(sbm).toarray(), beta_range=tau_range),
                                             graph.entropy(L=nx.laplacian_matrix(rand_graph_sparse).toarray(),
                                                           beta_range=tau_range)],
                               # ylabel=r'$\mathbf{S(\tau)}$',
                               ylabel=r'Von Neumann entropy',
                               tau_range=tau_range,
                               save_folder=spectral_ent_fold, fig_name='spectral_entropy_illustration.{:s}'.format(fig_format),
                               figsize=(7, 5),
                               curve_labels=labels,
                               show=args.show_fig)


    ############################ Plot density matrices corresponding to the SBM ###############################################################

    L = nx.laplacian_matrix(sbm).toarray()
    tau_range = [.5e-2, .8*1e-1, 1, 1e2]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 5*4+1))
    for i, tau in enumerate(tau_range):
        rho = expm(-tau*L)
        Z = np.trace(rho)
        rho = rho / Z # The illustration on the paper is not normalized
        print(Z, np.trace(rho))
        if i == 0:
            im = axes[i].imshow(rho, cmap='viridis', vmin=0, vmax=rho.max()/5)
            create_colorbar(fig=fig, ax=axes[i], mapp=im, array_of_values=rho/5, valfmt="{x:.3f}", position='right', )
        elif i == 1:
            im = axes[i].imshow(rho, cmap='viridis', vmin=0,
                                vmax=rho.max()/5)
                                # vmax=.1)
            create_colorbar(fig=fig, ax=axes[i], mapp=im,
                            array_of_values=[0, rho.max()/5],
                            # array_of_values=[0, .1],
                            valfmt="{x:.3f}",
                            position='right', )
        elif i == 2:
            im = axes[i].imshow(rho, cmap='viridis', vmin=0, vmax=.01)
            create_colorbar(fig=fig, ax=axes[i], mapp=im, array_of_values=[0, .01], valfmt="{x:.3f}", position='right', )
        elif i == 3:
            im = axes[i].imshow(rho, cmap='viridis', vmin=0, vmax=rho.max())
            create_colorbar(fig=fig, ax=axes[i], mapp=im, array_of_values=rho, valfmt="{x:.3f}", position='right',)
        axes[i].axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig('{:s}/rho_illustration.{:s}'.format(save_folder, args.fig_format), bbox_inches='tight', dpi=300)
    plt.show()

    # Eigenvalues
    # plot__eigenPdf(graph_list_grid=graph_list_grid,
    #                graph_list_random=graph_list_random,
    #                graph_list_gt=graph_list_gt,
    #                save_path=spectral_ent_fold,
    #                figname='eigenPdf.{:s}'.format(fig_format), curve_labels=labels, figsize=(6, 4), show=args.show_fig)
    # print(f'Figures saved to {spectral_ent_fold}')
    # a=0

