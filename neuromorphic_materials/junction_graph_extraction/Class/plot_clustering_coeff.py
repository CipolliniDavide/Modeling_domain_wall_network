#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:42:08 2021

@author: hp
"""


def plot_clustering_coeff(G, save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    cl_coeff = [nx.clustering(G, nodes=n) for n in G.nodes()]
    bins_edges = np.arange(0, 1, 10)
    plt.hist(cl_coeff, bins=bins_edges)
    plt.title(
        "%s. Mean %.3f clustering coefficient"
        % (save_path.split("/")[1], np.mean(cl_coeff))
    )
    plt.xticks(bins_edges)
    plt.xlabel("Clustering coefficient")
    plt.ylabel("# nodes")
    if save_path:
        plt.savefig(save_path + "clust_coeff.png")
    plt.close()
    print("Mean clustering coeff: %f" % np.mean(cl_coeff))
