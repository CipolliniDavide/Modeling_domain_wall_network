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

from scipy.stats import (levy, norm)
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from sklearn import mixture

from neuromorphic_materials.graph_scripts.helpers import visualize
from neuromorphic_materials.graph_scripts.helpers import utils, graph
from neuromorphic_materials.graph_scripts.helpers.visual_utils import set_ticks_label, create_order_of_magnitude_strings, set_legend
from neuromorphic_materials.graph_scripts.helpers.kmeans_tsne import plot_clustered_points, plot_k_means_tsn_points
from helpers.graph import (merge_nodes_by_distance, connected_components, filter_graphs_based_on_maxdegree)

def gaussian_mixture_compare(n_nodes_gt, n_nodes_vor, number_of_bins=[6, 8], name_fig= 'number_of_nodes', save_fold=None,
                             figsize=(6, 5), labelx='Nodes', labely='Probability Density',
                             random_seed=42, show=False, num_of_gaussians=1):
    np.random.seed(random_seed)
    all_data = [n_nodes_gt, n_nodes_vor]

    x = np.array(np.linspace(60, np.array(utils.unroll_nested_list(all_data)).max() + 20, 320)).reshape(-1, 1)
    temp = np.zeros(((2,) + np.shape(x)))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for p, (d, ax) in enumerate(zip(all_data, axes)):
        data = np.array(d).reshape(-1, 1)
        clf = mixture.GaussianMixture(n_components=num_of_gaussians, covariance_type='full', random_state=random_seed)
        clf.fit(data)
        weights = clf.weights_
        mean = clf.means_
        covs = clf.covariances_
        # print(mean)

        ax.hist(data.reshape(-1), density=True, color=colors[p], alpha=.5, bins=number_of_bins[p])
        col = ['C0', 'C1', 'C2']
        col = ['black', 'black']
        y_ax_sum = np.zeros_like(x)
        for i in range(len(weights)):
            y_axis0 = norm.pdf(x, float(mean[i][0]), np.sqrt(float(covs[i][0][0]))) * weights[i]  # 1st gaussian
            y_ax_sum = y_ax_sum + y_axis0
            ax.axvline(x=mean[i][0], color=col[i], linestyle='--', linewidth=1.5, alpha=.2)
            ax.plot(x, y_axis0, lw=1.5, c=col[i], ls='dashed', alpha=.2)
        temp[p] = y_ax_sum
        ax.plot(x, y_ax_sum, lw=4, c=colors[p], label=labels[p])
        if p==0:
            lab = labely
        else:
            lab = ''
        set_ticks_label(ax=ax, ax_type='x',
                        data=x,
                        num=7, valfmt="{x:.0f}",
                        ax_label='{:s}'.format(labelx))
        set_ticks_label(ax=ax, ax_type='y',
                        data=y_ax_sum,
                        num=4, valfmt="{x:.3f}",
                        ax_label=r'{:s}'.format(lab))
        set_legend(ax)

        # Set ticks for the secondary x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis
        mean_list = np.round(mean.reshape(-1), 0)
        ax2.set_xticks(mean_list)  # Add a single tick in the middle
        ax2.xaxis.tick_top()  # Position the tick on top
        # Set tick label color to gray
        set_ticks_label(ax=ax2, ax_type='x',
                        data=[0],
                        # add_ticks=[mean],
                        ticks=mean_list,
                        num=4, valfmt="{x:.1f}",
                        ax_label='')

        # tick_lab = ['{:.0f}'.format(m) if i == 0 else '{:.0f}'.format(m) for i, m in enumerate(mean_list)]
        tick_lab = [
            r'$\mu={:.0f},~(\sigma={:.0f})$'.format(m, np.sqrt(float(covs[i][0][0]))) if i == 0 else '{:.0f}'.format(m)
            for i, m in enumerate(mean_list)]
        tick_labels = ax2.get_xticklabels()
        # for xtick, color in zip(ax2.get_xticklabels(), colors):
        for xtick, color, tex in zip(tick_labels, colors, tick_lab):
            xtick.set_color(colors[p])
            xtick.set_text(tex)
        # Update the tick labels on the plot
        ax2.set_xticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig('{:s}/GMM{:d}_separ_{:s}.{:s}'.format(save_fold, int(num_of_gaussians), name_fig, fig_format))
    if show:
        plt.show()
    else:
        plt.close()

    #### Plot togheter
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    for i, y in enumerate(temp):
        ax.hist(all_data[i], density=True, color=colors[i], alpha=.2, bins=number_of_bins[i])
        ax.plot(x, y, c=colors[i], lw=8, label=labels[i])
    set_ticks_label(ax=ax, ax_type='x',
                    data=x,
                    num=7, valfmt="{x:.0f}",
                    ax_label='{:s}'.format(labelx))
    set_ticks_label(ax=ax, ax_type='y',
                    data=temp,
                    num=4, valfmt="{x:.3f}",
                    ax_label=r'{:s}'.format(labely))
    plt.tight_layout()
    set_legend(ax)
    plt.tight_layout()
    plt.savefig('{:s}/GMM{:d}_{:s}.{:s}'.format(save_fold, int(num_of_gaussians), name_fig, fig_format))
    if show:
        plt.show()
    else:
        plt.close()
    a=0


def histogram_compare(n_nodes_gt, n_nodes_vor, number_of_bins=6, name_fig= 'number_of_nodes', save_fold=None,
                      figsize = (6, 5), labelx='Nodes', labely='Probability Density', show=False):

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    # n_bins = [5, 5]
    temp_all = np.array(utils.unroll_nested_list([n_nodes_gt, n_nodes_vor]))
    n_bins = np.linspace(np.min(temp_all) - 3, np.max(temp_all), number_of_bins)
    # counts, n_bins = np.histogram(n_nodes_gt, 5)
    c_list= list()
    # hist_color_list = []
    for i, n in enumerate([n_nodes_gt, n_nodes_vor]):
        mypdf, _, bins = utils.empirical_pdf_and_cdf(sample=n, bins=n_bins)
        ax.plot(bins[1:], mypdf, alpha=1, color=colors[i], linewidth=8, label=labels[i],)
        count = mypdf

        mean = np.mean(n)
        std = np.std(n)
        # count, _, patches = ax.hist(n, bins=n_bins, density=True, label=labels[i], rwidth=0.8, color=colors[i], alpha=.5)
        # hist_color_list.append(patches[0].get_facecolor())
        ax.axvline(x=mean, color=colors[i], linestyle='--', linewidth=5)
        ax.axvspan(mean - std, mean + std, alpha=0.2, color=colors[i])
        c_list.append(count)
    # plt.title('# of nodes. Mean {:.4f} +- {:.4f}'.format(np.mean(n_nodes), np.std(n_nodes)))
    # set_ticks_label(ax=ax, ax_type='x',
    #                 data=temp_all,
    #                 ticks=np.linspace(temp_all.min(), temp_all.max(), 5, dtype=int),
    #                 num=3, valfmt="{x:d}",
    #                 ax_label='{:s}'.format(labelx))
    # set_ticks_label(ax=ax, ax_type='y',
    #                 data=utils.unroll_nested_list(c_list),
    #                 num=4, valfmt="{x:.2f}",
    #                 ax_label=r'{:s}'.format(labely))

    # Calculate bin widths and midpoints
    bin_width = n_bins[1] - n_bins[0]
    bin_midpoints = n_bins[:-1] + bin_width / 2
    set_ticks_label(ax=ax, ax_type='x',
                    data=n_bins,
                    # add_ticks=[mean],
                    ticks=bin_midpoints[::2],
                    num=4, valfmt="{x:.0f}",
                    ax_label='{:s}'.format(labelx))
    set_ticks_label(ax=ax, ax_type='y',
                    data=utils.unroll_nested_list(c_list),
                    num=3, valfmt="{x:.2f}",
                    ax_label=r'{:s}'.format(labely))
    # Create a secondary x-axis on top
    ax2 = plt.twiny()
    ax2.set_xlim(ax.get_xlim())  # Match the x-limits of the primary axis

    # Set ticks for the secondary x-axis
    mean_list = [np.mean(n) for n in [n_nodes_gt, n_nodes_vor]]
    ax2.set_xticks(mean_list)  # Add a single tick in the middle
    ax2.xaxis.tick_top()  # Position the tick on top

    # Set tick label color to gray
    set_ticks_label(ax=ax2, ax_type='x',
                    data=n_bins,
                    # add_ticks=[mean],
                    ticks=mean_list,
                    num=4, valfmt="{x:.1f}",
                    ax_label='')

    tick_lab = ['{:.1f}'.format(m) if i == 0 else '{:.1f}\n'.format(m) for i, m in enumerate(mean_list)]
    tick_labels = ax2.get_xticklabels()
    # for xtick, color in zip(ax2.get_xticklabels(), colors):
    for xtick, color, tex in zip(tick_labels, colors, tick_lab):
        xtick.set_color(color)
        xtick.set_text(tex)
    # Update the tick labels on the plot
    ax2.set_xticklabels(tick_labels)

    # ax2.tick_params(axis='x', colors=hist_color_list)

    set_legend(ax)
    plt.tight_layout()
    plt.savefig('{:s}/{:s}.{:s}'.format(save_fold, name_fig, fig_format))
    if show:
        plt.show()
    else:
        plt.close()

def create_lists_of_graph_prop(graph_list):
    # Create lists
    n_nodes = list()
    n_edges = list()
    deg_list = list()
    clustering_list = list()

    for g in graph_list:
        n_nodes.append(g.number_of_nodes())
        n_edges.append(g.number_of_edges())
        deg_list.append([d for n, d in g.degree()])
        clustering_list.append(graph.compute_clustering_coefficient_distribution(g))

    return n_nodes, n_edges, utils.unroll_nested_list(deg_list), utils.unroll_nested_list(clustering_list)

def plot_degree_correlation_function_AllTogether(graph_list_of_list, save_path=None, figname=None, figsize=(10,10),
                                                 show=False,
                                                 log=False):
    deg_range_list = []
    corr_l = []
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=figsize)
    for i, graph_list in enumerate(graph_list_of_list):
        degree_range, degree_correlation = graph.degree_correlation_function(graph_list, separate_matrices=False)
        degree_range = degree_range[~np.isnan(degree_correlation)]
        degree_correlation = degree_correlation[~np.isnan(degree_correlation)]
        corr_l.append(degree_correlation)
        deg_range_list.append(degree_range)

        # ax.scatter(degree_range, degree_correlation, marker='o', linestyle='-', color=colors[i], label=labels[i])
        ax.plot(degree_range, degree_correlation, linestyle='-', linewidth=6, color=colors[i], label=labels[i])

    corr_l = np.array(utils.unroll_nested_list(corr_l))
    deg_range_list = np.unique(np.array(utils.unroll_nested_list(deg_range_list)))
    if len(deg_range_list) < 7:
        num=6
    else: num=3
    set_ticks_label(ax=ax, ax_type='x',
                    # data=[np.min([np.min(d) for d in eigen_list]), np.max([np.max(d) for d in eigen_list])],
                    data=corr_l,
                    ticks=np.linspace(deg_range_list.min(), deg_range_list.max(), num, dtype=int),
                    num=num, valfmt="{x:.0f}",
                    only_ticks=False, #tick_lab=[r'$\mathbf{10^{-1}}$', r'$\mathbf{10^{0}}$', r'$\mathbf{10^{-2}}$'],
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label=r'k', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])

    set_ticks_label(ax=ax, ax_type='y', data=corr_l, num=4, valfmt="{x:.1f}",
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label=r"$\mathbf{k_{nn}(k)}$",
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])
    # # plt.set_ylim((min_d, max_d))
    set_legend(ax)
    if log:
        plt.yscale('log')
        plt.xscale('log')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path+figname)
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_together_of_clust_vs_degree(deg_list_list, clust_list_list, save_path=None, figname=None, figsize=(8,8),
                                         show=False, log=False, labely="C(k)"):

    fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True, sharex=True, figsize=figsize)
    for i, (deg_l, clu_l) in enumerate(zip(deg_list_list, clust_list_list)):
        deg_unique, mean_cl, std_cl = graph.degree_clustering_mean_std(degrees=np.array(deg_l),
                                                                       clustering_coefficients=np.array(clu_l))
        ax.plot(deg_unique, mean_cl, linewidth=6, label=labels[i], c=colors[i])
        ax.fill_between(deg_unique, mean_cl - std_cl/2, mean_cl + std_cl/2, alpha=0.2,
                         color=colors[i])
        ax.axhline(y=np.mean(clu_l), color=colors[i], linestyle='--')
        # ax.set_title('{:s}'.format(crop_name[index]))

    set_ticks_label(ax=ax, ax_type='x', data=[np.min([np.min(d) for d in deg_list_list]),
                                              np.max([np.max(d) for d in deg_list_list])],
                    num=4, valfmt="{x:.0f}",
                    ticks=np.unique(deg_unique), only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label='k', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])

    if log:
        font_properties = {'weight': 'bold', 'size': 'xx-large'}
        ax.set_yscale('log')
        ax.set_ylabel(labely)
        y_label = ax.yaxis.get_label()
        y_label.set_font_properties(font_properties)
        y_tick_labels = ax.get_yticklabels()
        for label in y_tick_labels:
            label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
    else:
        set_ticks_label(ax=ax, ax_type='y', data=mean_cl, num=4, valfmt="{x:.2f}",
                        fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                        ax_label=labely,
                        fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                        scale=None, add_ticks=[])

    plt.tight_layout()
    set_legend(ax=ax, loc=4)
    if save_path is not None:
        plt.savefig(save_path+figname)
    if show:
        plt.show()




def plot_mean_degree_(deg_list, save_fold=None, save_name='mean_degree_std.png', figsize=(8, 4), show=False):
    mean_degree = np.array([np.mean(d) for d in deg_list])
    std_degree = np.array([np.std(d) for d in deg_list])

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    ax1.errorbar(x=np.arange(len(deg_list)), y=mean_degree, yerr=std_degree, marker='o')
    set_ticks_label(ax=ax1, ax_type='x', data=np.arange(len(deg_list)), num=5, valfmt="{x:.0f}",
                    ticks=None, only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label='Id sample', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])
    count, bins, bar = ax2.hist(mean_degree, orientation='horizontal', density=True)
    set_ticks_label(ax=ax1, ax_type='y', data=[np.max(mean_degree + std_degree), np.min(mean_degree - std_degree)],
                    num=5, valfmt="{x:.2f}",
                    ticks=None, only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label='<k>', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])
    set_ticks_label(ax=ax2, ax_type='x', data=count, num=4, valfmt="{x:.0f}",
                    ticks=None, only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'},
                    ax_label='# counts', fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'},
                    scale=None, add_ticks=[])
    plt.suptitle('Mean degree <k> of groundTruths')
    plt.tight_layout()
    if save_fold:
        plt.savefig(save_fold + save_name)
    if show:
        plt.show()
    else: plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    # flat_list = [item for sublist in deg_list for item in sublist]
    deg_unique, counts = np.unique(deg_list, return_counts=True)
    # deg_unique = np.insert(deg_unique, 0, 0)
    # counts = np.insert(counts, 0, 0)



    def levy_curve_fit(x, y):
        """
        Fit a Levy distribution curve to the given x and y values.

        Parameters:
            x (numpy.ndarray): Array of x values.
            y (numpy.ndarray): Array of y values.

        Returns:
            fitted_params (tuple): Fitted parameters (location, scale) for the Levy distribution.
        """
        # Interpolate the data to generate more points
        x_interp = np.linspace(min(x), max(x), num_points)
        y_interp = np.interp(x_interp, x, y)

        def levy_distribution(x, loc, scale):
            return levy.pdf(x, loc, scale)

        fitted_params, _ = curve_fit(levy_distribution, x_interp, y_interp)
        return fitted_params

    # fitted_params = levy_curve_fit(x=deg_unique, y=counts / counts.sum())
    # ax.plot(deg_unique, levy.pdf(deg_unique, *fitted_params), color='red', label="Fitted Levy Distribution")

    ax.bar(deg_unique, counts / counts.sum())
    ax.set_title('All 16 crops together')
    set_ticks_label(ax=ax, ax_type='x', data=deg_unique, valfmt="{x:.0f}", ticks=deg_unique, ax_label='k')
    set_ticks_label(ax=ax, ax_type='y', data=counts / counts.sum(), valfmt="{x:.1f}", ax_label='P(k)')
    plt.tight_layout()
    if save_fold:
        plt.savefig(save_fold + 'histogram_all16Crops.{:s}'.format(fig_format))
    if show:
        plt.show()
    else:
        plt.close()
    a=0


def compare_bar_plot(deg_list_gt, deg_list_vor, save_fold=None, save_name='mean_degree_std.png',
                     colors=['blue', 'red'], labels=['BFO', 'Voronoi'],
                     figsize = (6, 5), labelx='Nodes', labely='Probability', show=False):

    fig, ax = plt.subplots(figsize=figsize)
    deg_un_list=[]
    c_l = []
    for i, deg in enumerate([deg_list_gt, deg_list_vor]):
        # flat_list = [item for sublist in deg for item in sublist]
        deg_unique, counts = np.unique(deg, return_counts=True)
        deg_un_list.append(deg_unique)
        c_l.append(counts / counts.sum())
        # print(np.sum(counts / counts.sum()))
        # ax.bar(deg_unique, counts / counts.sum(), color=colors[i], label=labels[i], alpha=.6)
        ax.plot(deg_unique, counts / counts.sum(), color=colors[i], label=labels[i], linewidth=8)

    # ax.set_title('All 16 crops together')
    deg_un_list = np.unique(utils.unroll_nested_list(deg_un_list))
    set_ticks_label(ax=ax, ax_type='x', data=deg_un_list, valfmt="{x:.0f}", ticks=deg_un_list,
                    ax_label=labelx)
    # set_ticks_label(ax=ax, ax_type='y', data=utils.unroll_nested_list(c_l), valfmt="{x:.1f}", ax_label=labely)
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    ax.set_yscale('log')
    ax.set_ylabel(labely)
    y_label = ax.yaxis.get_label()
    y_label.set_font_properties(font_properties)
    y_tick_labels = ax.get_yticklabels()
    for label in y_tick_labels:
        label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
    set_legend(ax=ax)
    plt.tight_layout()

    if save_fold:
        plt.savefig(save_fold + 'histogram_Pk.{:s}'.format(fig_format))
    if show:
        plt.show()
    else:
        plt.close()
    a=0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp_GT', '--load_path_GT', default='../../Dataset/GroundTruth2/graphml', type=str)
    parser.add_argument('-lp_VOR', '--load_path_Vor', default='../../Dataset/Voronoi128_d0.38_p0.15_beta2.99/graphml', type=str)
    parser.add_argument('-svp', '--save_path', default=None, type=str)
    parser.add_argument('-eps', '--merge_epsilon', default=1.4, type=float)
    parser.add_argument('-figform', '--fig_format', default='.pdf', type=str)
    parser.add_argument('-show', '--show_fig', default=False, type=bool)
    args = parser.parse_args()

    fig_format = args.fig_format
    merge_epsilon = args.merge_epsilon

    labels = ['BFO', 'Voronoi']
    colors = ['blue', 'red']

    if args.save_path is None:
        root = os.getcwd().rsplit('/', 2)[0] + '/'
        save_folder = root + 'Figures/StatAnalysis/GT_Vor{:s}/eps{:.1f}'.format(args.load_path_Vor.split('Voronoi')[1].split('/')[0], merge_epsilon)
    else:
        save_folder = '{:s}/eps{:.1f}'.format(args.save_path, merge_epsilon)

    # Load list of file names
    file_list_gt = sorted(glob('{:s}/*.graphml'.format(args.load_path_GT)))

    file_list_vor = sorted(glob('{:s}/*.graphml'.format(args.load_path_Vor))) #[:300]

    # Load graphs
    graph_list_gt = list()
    graph_list_vor = list()
    crop_name = list()

    for i, fgt in enumerate(file_list_gt):
        crop_name.append(fgt.rsplit('/')[-1].split('.')[0])
        graph_list_gt.append(connected_components(nx.read_graphml(fgt))[0])
        # graph_list_gt.append(connected_components(merge_nodes_by_distance(nx.read_graphml(fgt), epsilon=merge_epsilon, return_removed_nodes=False))[0])
    for i, fvor in enumerate(file_list_vor):
        graph_list_vor.append(connected_components(merge_nodes_by_distance(nx.read_graphml(fvor),
                                                                           epsilon=merge_epsilon,
                                                                           return_removed_nodes=False))[0])

    graph_list_vor, perc = filter_graphs_based_on_maxdegree(graph_list=graph_list_vor, degree_threshold=7,
                                                            return_percentage=True)
    print(f'Percentage of removed graphs: {perc}%. Fraction of used graphs: {len(graph_list_vor)}/{len(file_list_vor)}')

    save_deg_analysis = '{:s}/DegreeAnalysis/'.format(save_folder)
    save_node_edge_analysis = '{:s}/CountNodesEdges/'.format(save_folder)
    utils.ensure_dir(save_deg_analysis)
    utils.ensure_dir(save_node_edge_analysis)


    n_nodes_gt, n_edges_gt, deg_list_gt, clustering_list_gt = create_lists_of_graph_prop(graph_list_gt)
    n_nodes_vor, n_edges_vor, deg_list_vor, clustering_list_vor = create_lists_of_graph_prop(graph_list_vor)

    ########################## Fig 7A, B:  histograms of nodes and edges in esambles ###################################
    for i in range(1, 2):
        gaussian_mixture_compare(n_nodes_gt, n_nodes_vor, number_of_bins=[6, 15], name_fig=f'number_of_nodes',
                          save_fold=save_node_edge_analysis, show=args.show_fig, num_of_gaussians=i,
                          figsize=(7, 5), labelx='Nodes', labely='Probability', random_seed=13)
        gaussian_mixture_compare(n_edges_gt, n_edges_vor, number_of_bins=[6, 15], name_fig=f'number_of_edges',
                                 save_fold=save_node_edge_analysis, show=args.show_fig, num_of_gaussians=i,
                                 figsize=(7, 5), labelx='Edges', labely='Probability', random_seed=13)
    print('Figures about number of nodes and edges saved to:\n\t{}\n'.format(save_node_edge_analysis))

    # Histograms of number of edges and nodes
    # histogram_compare(n_nodes_gt, n_nodes_vor, number_of_bins=12, name_fig='number_of_nodes',
    #                   save_fold=save_node_edge_analysis, show=args.show_fig,
    #                   figsize=(7, 5), labelx='Nodes', labely='Probability')
    # histogram_compare(n_edges_gt, n_edges_vor, number_of_bins=12, name_fig='number_of_edges',
    #                   save_fold=save_node_edge_analysis, show=args.show_fig,
    #                   figsize=(7, 5), labelx='Edges', labely='Probability')


    ############################### Fig 7C: degree distribution ########################################################
    compare_bar_plot(deg_list_gt=deg_list_gt, deg_list_vor=deg_list_vor, save_name='mean_degree_std.{:s}'.format(fig_format),
                     figsize=(6, 5), labelx='k', labely='P(k)', show=args.show_fig, save_fold=save_deg_analysis)
    # for i, f in enumerate(file_list):
    #     visualize.degree_analysis(graph_list[i], save_name='deg{:d}'.format(i), save_fold=save_deg_analysis,
    #                               title="<k>+-std = {:.4f}+-{:.4f}".format(np.mean(deg_list[i]), np.std(deg_list[i])))

    ###################   ######### Fig 7D: clustering coeff. vs degree  ###############################################
    plot_all_together_of_clust_vs_degree(deg_list_list=[deg_list_gt, deg_list_vor],
                                         clust_list_list=[clustering_list_gt, clustering_list_vor],
                                         save_path=save_deg_analysis, figsize=(5, 4), log=False,
                                         figname='allinOne_clust_vs_deg.{:s}'.format(fig_format), show=args.show_fig)

    ######################################## Degree correlations #######################################################
    # TODO: to be checked... not included in the paper
    # plot_degree_correlation_function_AllTogether(graph_list_of_list=[graph_list_gt, graph_list_vor],
    #                                              save_path=save_deg_analysis,
    #                                              figname='deg_corr_funcAll.{:s}'.format(fig_format), figsize=(6, 4),
    #                                              show=args.show_fig, log=False)

    # corr_ = graph.compute_degree_correlation_matrix(graph_list_vor, separate_matrices=False)
    # plt.imshow(corr_, interpolation='bilinear')
    # if args.show_fig:
    #     plt.show()
    # else:
    #     plt.close()

    # plot_mean_degree_(deg_list_vor, save_fold=save_deg_analysis, show=args.show_fig)

    #################################### End ###########################################################################
    print(f'Figures degree analysis saved to :\n\t{save_deg_analysis}\nand to:\n\t{save_node_edge_analysis}\n')