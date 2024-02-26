#! /usr/bin/env python3
import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd().rsplit('/', 1)[0]))

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


def eigen_distribution(graph_list_gt, bins):
    eigen_list_gt = list()

    for gt in graph_list_gt:
        eigen_list_gt.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))

    flattened_list_gt = np.clip([item for sublist in eigen_list_gt for item in sublist], 0, 1e15)

    if bins is None:
        bins=80
    # Compute PDF using kernel density estimation (KDE)
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_gt, bins=bins)
    return mypdf, bins

def specific_heat2(graph_list_gt, tau_range):
    eigen_list_gt = list()
    if graph_list_gt is not list:
        graph_list_gt_temp = [graph_list_gt]
    for gt in graph_list_gt_temp:
        eigen_list_gt.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))

    flattened_list_eigen = [item for sublist in eigen_list_gt for item in sublist]
    flattened_list_eigen = np.clip(flattened_list_eigen, 0, np.max(flattened_list_eigen))
    pseudo_rho = np.array([np.exp(-flattened_list_eigen*tau) for tau in tau_range])
    Z = pseudo_rho.sum()
    mean_lambda = (flattened_list_eigen * pseudo_rho).sum(axis=1) / Z
    spec_heat = - np.gradient(mean_lambda, tau_range, edge_order=2) * tau_range**2
    # plt.semilogx(tau_range, spec_heat)
    # plt.show()
    return spec_heat


def specific_heat(S, tau_range):
    # Calculate the first derivative of spec_entropy with respect to time , edge_order=2
    spec_heat = - np.gradient(S, tau_range, edge_order=2) * tau_range #* \
                # np.log(torch.clip(adj.sum(1), min=0, max=1).sum(1)).detach().numpy()
    # plt.semilogx(tau_range, S)
    # plt.semilogx(tau_range, spec_heat, '--')
    # plt.show()
    return np.clip(spec_heat, 0, spec_heat.max())

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


def plot_overlapped_SE_MeanSTD_next_to_eigen(list_ofList_of_graphs,
                                             # list_of_spec_entropy,
                                             # specEnt_list_gt, specEnt_list_rand, specEnt_list_grid
                                             tau_range=None,
                                             reference_list_of_graphs=None,
                                             curve_labels = ['2D-Grid', 'Erdős–Rényi', 'Small-world'],
                                             colors = ['orange', 'purple', 'green'],
                                             alpha=.7,
                                             figsize=(6,4),
                                             save_folder=None, fig_name=None, show=False,
                                             ylabel="S",
                                             eigen_bins=40):

    se_ref = []
    for g in reference_list_of_graphs:
        num_components = nx.number_connected_components(g)
        s = graph.entropy(L=nx.laplacian_matrix(g).toarray(), beta_range=tau_range)
        # se_ref.append(
        #     1 / (np.log(g.number_of_nodes()) - np.log(num_components)) * (s - np.log(num_components)))
        se_ref.append(s / np.log(g.number_of_nodes()))

    eigen_ref, bins_ref = eigen_distribution(reference_list_of_graphs, bins=eigen_bins)

    spec_heat = []
    fontdict_label = {'weight': 'bold', 'size': 20}
    font_dict_tick = {'weight': 'bold', 'size': 'xx-large'}

    fig, axes = plt.subplots(ncols=3, nrows=len(list_ofList_of_graphs),
                             figsize=((5*3)+3, len(list_ofList_of_graphs)*4 +1),
                             sharex=False, sharey=False)
    for i, list_of_g in enumerate(list_ofList_of_graphs):

        if 'Grid' in curve_labels[i]:
            print('grid')
            pos = {(i, j): (j, -i) for i, j in list_of_g[0].nodes()}
        else:
            pos = nx.spring_layout(list_of_g[0])
        # nx.draw_networkx(G=connected_components(list_of_g[0])[0], pos=pos, ax=axes[i, 0], with_labels=False,
                         # node_color=colors[i], width=5, alpha=.5)
        nx.draw_networkx(G=list_of_g[0], pos=pos, ax=axes[i, 0], with_labels=False,
                         node_color=colors[i], width=5, alpha=.5)
        axes[i, 0].set_axis_off()
        # Compute the spectral_ent
        se = []
        for g in list_of_g:
            num_components = nx.number_connected_components(g)
            s = graph.entropy(L=nx.laplacian_matrix(g).toarray(), beta_range=tau_range)
            # se.append(1 / (np.log(g.number_of_nodes()) - np.log(num_components)) * (s - np.log(num_components)))
            se.append(s/np.log(g.number_of_nodes()))

        # Calculate mean and standard deviation across curves
        mean_curve = np.mean(se, axis=0)
        std_curve = np.std(se, axis=0)

        mean_curve_ref = np.mean(se_ref, axis=0)
        std_curve_ref = np.std(se_ref, axis=0)

        # Create the plot
        ax = axes[i, 1]

        ax.semilogx(tau_range, mean_curve, linewidth=4, label=curve_labels[i], color=colors[i], alpha=alpha)
        ax.fill_between(tau_range, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=colors[i])

        ax.semilogx(tau_range, mean_curve_ref, linewidth=4, label='BFO', color='blue', alpha=alpha)
        ax.fill_between(tau_range, mean_curve_ref - std_curve_ref, mean_curve_ref + std_curve_ref, alpha=0.2,
                        color=colors[0])
        # ax.set_xscale('log')
        ax.set_ylim((0, 1.05))

        set_ticks_label(ax=ax, ax_type='y', data=[0, 1],
                        # data=[np.min(specEnt_list_gt), np.min(specEnt_list_rand),
                        #                           np.max(specEnt_list_gt), np.max(specEnt_list_rand)],
                        num=4, valfmt="{x:.1f}",
                        ax_label=ylabel,
                        fontdict_ticks_label=font_dict_tick,
                        fontdict_label=fontdict_label
                        )

        x_tick_labels = ax.get_xticklabels()
        for k, label in enumerate(x_tick_labels):
            if k % 2 == 0:
                label.set_visible(False)
            else:
                label.set_font_properties(font_dict_tick)
        ax.set_xlabel(r'$\mathbf{\tau}$', fontdict=font_dict_tick)

        # Set font properties for the y-axis label
        y_label = ax.yaxis.get_label()
        y_label.set_font_properties(fontdict_label)
        x_label = ax.xaxis.get_label()
        x_label.set_font_properties(fontdict_label)
        ax.tick_params(axis='both', which='both', width=1.7, length=7.5)
        ax.tick_params(axis='both', which='minor', width=1.7, length=4)
        set_legend(ax, title='', ncol=1, loc=0)

        ########################### Specific heat on right axis #######################################################
        # ax2 = ax.twinx()
        # spec_heat.append(np.log(mean_numb_nodes)*specific_heat(S=mean_curve, tau_range=tau_range))
        #
        # ax2.semilogx(tau_range, np.log(mean_numb_nodes)*specific_heat(S=mean_curve, tau_range=tau_range), '--', color=colors[i])
        # ax2.semilogx(tau_range, np.log(mean_numb_nodes)*specific_heat(S=mean_curve_ref, tau_range=tau_range), '--', color='blue')
        #
        # set_ticks_label(ax=ax2, ax_type='y', data=np.array(spec_heat).reshape(-1),
        #                 num=4, valfmt="{x:.1f}",
        #                 ax_label='ClogN',
        #                 fontdict_ticks_label=font_dict_tick,
        #                 fontdict_label=fontdict_label
        #                 )
        # ax2.set_ylim((1e-1, 3))
        # ax2.set_yscale('log')

        ################################# Eigenvalue distribution ######################################################
        eigen_pdf, bins = eigen_distribution(list_of_g, bins_ref)

        ax_eigen = axes[i, 2]
        if 'Grid' in curve_labels[i]:
            ind = eigen_pdf > 1e-4
            ax_eigen.plot(bins[1:][ind], eigen_pdf[ind], alpha=alpha, linewidth=4, label=curve_labels[i], color=colors[i])
        else:
            ax_eigen.plot(bins[1:], eigen_pdf, alpha=alpha, linewidth=4, label=curve_labels[i], color=colors[i])
        ax_eigen.plot(bins[1:], eigen_ref, alpha=alpha, linewidth=4, label='BFO', color='blue')

        ax_eigen.set_yscale('log')
        ax_eigen.set_xscale('log')
        # plt.show()
        y_tick_labels = ax_eigen.get_yticklabels()
        for label in y_tick_labels:
            label.set_font_properties(font_dict_tick)
        x_tick_labels = ax_eigen.get_xticklabels()
        for label in x_tick_labels:
            label.set_font_properties(font_dict_tick)

        ax_eigen.set_xlabel(r'$\mathbf{\lambda}$')
        ax_eigen.set_ylabel(r'$\mathbf{P(\lambda})$')
        # Set font properties for the y-axis label
        font_properties = {'weight': 'bold', 'size': 'xx-large'}
        y_label = ax_eigen.yaxis.get_label()
        y_label.set_font_properties(fontdict_label)
        x_label = ax_eigen.xaxis.get_label()
        x_label.set_font_properties(fontdict_label)
        ax_eigen.tick_params(axis='both', which='both', width=2, length=7.5)
        ax_eigen.tick_params(axis='both', which='minor', width=2, length=4)
        set_legend(ax_eigen, title='', ncol=1, loc=0)

    plt.tight_layout()
    if save_folder:
        plt.savefig(save_folder + fig_name)
    if show:
        plt.show()
    else:
        plt.close()
    a=0


def plot_overlapped_SE_MeanSTD(specEnt_list_gt, specEnt_list_rand, specEnt_list_grid, tau_range,
                               curve_labels=['BFO', 'Grid', 'Erdős–Rényi'], alpha=.7,
                               figsize=(6,4), save_folder=None, fig_name=None, show=False, ylabel="S"):
    # log_min = -3
    # log_max = 4
    # ticks = np.logspace(start=log_min, stop=log_max, num=4, endpoint=True)
    # ticks = [10e-3, .1, 10e2, 10e4]
    # ticks_lab = [r'$\mathbf{10^{-3}}$', r'$\mathbf{10^{-1}}$', r'$\mathbf{10^{1}}$', r'$\mathbf{10^{4}}$']
    # One plot Spectral Entropy
    spec_heat = []
    # figsize = (4, 8)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, sharex=True, sharey=True)
    ax2 = ax.twinx()
    for i, se in enumerate([specEnt_list_gt, specEnt_list_grid, specEnt_list_rand]):
        # ax = axes[i]
        # Calculate mean and standard deviation across curves
        mean_curve = np.mean(se, axis=0)
        std_curve = np.std(se, axis=0)
        # Create the plot
        ax.semilogx(tau_range, mean_curve, linewidth=4, label=curve_labels[i], color=colors[i], alpha=alpha)
        ax.fill_between(tau_range, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=colors[i])
        spec_heat.append(specific_heat(S=mean_curve, tau_range=tau_range))
        ax2.plot(tau_range, specific_heat(S=mean_curve, tau_range=tau_range), '--', color=colors[i])


    # ax2.grid()
    # for ax in [ax1, ax2]:
    #     set_ticks_label(ax=ax, ax_type='x', data=[log_min, log_max],
    #                     num=4, valfmt="{x:s}", ticks=ticks,
    #                     only_ticks=False, tick_lab=ticks_lab,
    #                     ax_label=r'$\mathbf{\tau}$', scale='log')
    #
    set_ticks_label(ax=ax2, ax_type='y', data=np.array(spec_heat).reshape(-1),
                    num=4, valfmt="{x:.1f}",
                    ax_label='C',
                    )

    set_ticks_label(ax=ax, ax_type='y', data=[np.min(specEnt_list_gt), np.min(specEnt_list_rand),
                                              np.max(specEnt_list_gt), np.max(specEnt_list_rand)],
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


def set_ticks_and_labels_sizes(ax,
                               font_properties_lab = {'weight': 'bold', 'size': 'xx-large'},
                               font_properties_ticks = {'weight': 'bold', 'size': 'xx-large'}):

    y_tick_labels = ax.get_yticklabels(which='both')
    for label in y_tick_labels:
        label.set_font_properties(font_properties_ticks)
    x_tick_labels = ax.get_xticklabels()
    for label in x_tick_labels:
        label.set_font_properties(font_properties_ticks)

    y_label = ax.yaxis.get_label()
    y_label.set_font_properties(font_properties_lab)
    x_label = ax.xaxis.get_label()
    x_label.set_font_properties(font_properties_lab)
    ax.tick_params(axis='both', which='both', width=2, length=7.5)
    ax.tick_params(axis='both', which='minor', width=2, length=4)


def plot__eigenPdf(graph_list_gt, graph_list_grid, graph_list_random, save_path=None,
                   curve_labels=['Grid', 'BFO', 'Erdős–Rényi'],
                   figname=None, figsize=(6,4), show=False, alpha=.8):

    eigen_list_grid = list()
    eigen_list_gt = list()
    eigen_list_random = list()

    for gt in graph_list_gt:
        eigen_list_gt.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))
    for vor in graph_list_grid:
        eigen_list_grid.append(eigh(nx.laplacian_matrix(vor).toarray(), eigvals_only=True))
    for gt in graph_list_random:
        eigen_list_random.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))

    flattened_list_rand = np.clip([item for sublist in eigen_list_random for item in sublist], 0, 1e15)
    flattened_list_gt = np.clip([item for sublist in eigen_list_gt for item in sublist], 0, 1e15)
    flattened_list_vor = np.clip([item for sublist in eigen_list_grid for item in sublist], 0, 1e15)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    # Compute PDF using kernel density estimation (KDE)
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_gt, bins=80)

    ind = mypdf > 0
    ind[-3:] = True
    ax.plot(bins[1:][ind], mypdf[ind], alpha=alpha, linewidth=4, label=curve_labels[0], color=colors[0])
    # ax.plot(bins[1:], mypdf, alpha=1, linewidth=4, label=curve_labels[0], color=colors[0])
    # plt.show()
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_vor, bins=bins)
    ax.plot(bins[1:], mypdf, alpha=alpha, linewidth=4, label=curve_labels[1], color=colors[1])
    mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=flattened_list_rand, bins=bins)
    ax.plot(bins[1:], mypdf, alpha=alpha, linewidth=4, label=curve_labels[2], color=colors[2])

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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp_GT', '--load_path_GT', type=str,
                        default='/home/davide/PycharmProjects/Bfo_network/Dataset/GroundTruth2/graphml',
                        )
    parser.add_argument('-lp_VOR', '--load_path_Vor', type=str,
                        default='/home/davide/PycharmProjects/Bfo_network//Dataset/Voronoi128_d0.32_p0.20_beta2.95/graphml')
    parser.add_argument('-svp', '--save_path', default=None, type=str)
    parser.add_argument('-eps', '--merge_epsilon', default=1.4, type=float)
    parser.add_argument('-figform', '--fig_format', default='svg', type=str)
    parser.add_argument('-show', '--show_fig', default=True, type=bool)
    args = parser.parse_args()

    fig_format = args.fig_format
    merge_epsilon = args.merge_epsilon

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
    graph_list_grid = list()
    crop_name = list()

    for i, fgt in enumerate(file_list_gt):
        crop_name.append(fgt.rsplit('/')[-1].split('.')[0])
        graph_list_gt.append(connected_components(nx.read_graphml(fgt))[0])
    # for i, fvor in enumerate(file_list_vor):
    #     graph_list_grid.append(connected_components(merge_nodes_by_distance(nx.read_graphml(fvor), epsilon=merge_epsilon, return_removed_nodes=False))[0])

    # mean_numb_nodes = np.mean([g.number_of_nodes() for g in graph_list_grid])
    mean_numb_nodes = 108
    mean_numb_edges = 143
    graph_list_grid = [nx.grid_2d_graph(m=int(np.sqrt(mean_numb_nodes)), n=int(np.sqrt(mean_numb_nodes))) for _ in range(1)]
    graph_list_random = [nx.erdos_renyi_graph(n=mean_numb_nodes, p=2*mean_numb_edges/(mean_numb_nodes*(mean_numb_nodes-1))) for _ in range(100)]
    graph_list_small_world = [nx.connected_watts_strogatz_graph(n=mean_numb_nodes, k=4, p=.03, tries=100, seed=None) for _ in range(100)]
    # graph_list_random = [nx.random_reference(G, niter=100, connectivity=True, seed=1) for G in graph_list_gt]
    # Create lists
    spectral_ent_gt = list()
    spectral_ent_grid = list()
    spectral_ent_rand = list()

    log_min = -3  # Start value (10^-3)
    log_max = 4  # Stop value (10^4)
    num_points = 300  # Number of points

    # Generate the log-spaced array
    tau_range = np.logspace(log_min, log_max, num=num_points)

    # Spectral entropy #####################################
    spectral_ent_fold = save_folder + '/SE_RandN_GRID/'
    utils.ensure_dir(spectral_ent_fold)
    labels = ['Erdős–Rényi', 'Small-world', '2D-Grid']
    colors = ['darkorange', 'green', 'brown']
    mean_numb_nodes = 108
    mean_numb_edges = 143
    graph_list_random = [connected_components(nx.erdos_renyi_graph(n=mean_numb_nodes,
                                                                   p=2 * mean_numb_edges / (mean_numb_nodes * (mean_numb_nodes - 1))))[0]
                         for _ in range(100)]
    graph_list_small_world = [nx.connected_watts_strogatz_graph(n=mean_numb_nodes, k=4, p=.05, tries=100, seed=None)
                              for _ in range(100)]
    graph_list_grid = [nx.grid_2d_graph(m=int(np.sqrt(mean_numb_nodes)), n=int(np.sqrt(mean_numb_nodes))) for
                       _ in
                       range(1)]
    plot_overlapped_SE_MeanSTD_next_to_eigen(list_ofList_of_graphs=[graph_list_random,
                                                                    graph_list_small_world,
                                                                    graph_list_grid],
                                             eigen_bins=50,
                                             reference_list_of_graphs=graph_list_gt,
                                             # ylabel=r'$\mathbf{S(\tau)}$',
                                             curve_labels= labels, colors=colors,
                                             ylabel=r'$\mathbf{S}$',
                                             tau_range=tau_range,
                                             alpha=.7,
                                             save_folder=spectral_ent_fold,
                                             fig_name='overlapped_SE_with_eigen.{:s}'.format(fig_format),
                                             figsize=(7, 5),
                                             show=True,)
                                             # show=args.show_fig)

    # plot_overlapped_SE_MeanSTD(specEnt_list_gt=spectral_ent_gt, specEnt_list_rand=spectral_ent_rand,
    #                            specEnt_list_grid = spectral_ent_grid,
    #                            # ylabel=r'$\mathbf{S(\tau)}$',
    #                            ylabel=r'$\mathbf{S}$',
    #                            tau_range=tau_range,
    #                            alpha=.7,
    #                            save_folder=spectral_ent_fold, fig_name='overlapped_SE.{:s}'.format(fig_format),
    #                            figsize=(7, 5), curve_labels=labels, show=args.show_fig)
    #
    #
    # plot__eigenPdf(graph_list_grid=graph_list_grid,
    #                graph_list_random=graph_list_random,
    #                graph_list_gt=graph_list_gt,
    #                save_path=spectral_ent_fold,
    #                alpha=.7,
    #                figname='eigenPdf.{:s}'.format(fig_format), curve_labels=labels, figsize=(6, 4), show=args.show_fig)
    # print(f'Figures saved to {spectral_ent_fold}')
    # a=0

