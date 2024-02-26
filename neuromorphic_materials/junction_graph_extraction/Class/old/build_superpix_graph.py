#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 17:38:46 2021

@author: hp
"""

import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage import io
from skimage.color import gray2rgb
from skimage.future import graph
from skimage.segmentation import slic

# from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE

sys.path.extend(["/home/hp/PycharmProjects/graph/scripts"])
from Class.superpix_graph import add_attr_, rag, remove_attribute, superpix_graph
from Class.utils import utils
from Class.visualize import visualize

for numSegments in [50, 100, 150]:  # 200, 300, 500]:
    # loop over the number of segments
    image = io.imread(
        "/home/hp/Scaricati/results/elongation/JR38_CoCr_6_70000.ibw_4_12_2_3.pgm_.png"
    )
    # numSegments=10
    sigma = 5
    save_name = "JR38_elongation"
    roothpath = "/home/hp/PycharmProjects/graph/"
    save_fold = roothpath + "superpix-rag_mean_color/{}segments_sigma{}/{}/".format(
        numSegments, sigma, save_name
    )
    utils.ensure_dir(save_fold)

    # angles, _ = utils.angles_with_sobel_filter(image)

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    image = gray2rgb(image)
    # Build graph with SLIC
    segments = slic(
        image, n_segments=numSegments, sigma=sigma, start_label=1, convert2lab=True
    )
    visualize.plot_superpix_image(image, segments, save_fold=save_fold)
    # segments = slic(image, compactness=30, n_segments=numSegments, start_label=1)
    G = superpix_graph(image=image, segments=segments).create_graph()
    G = graph.rag_mean_color(image, segments, connectivity=2)
    fig, ax = plt.subplots(nrows=1, figsize=(18, 16))
    lc = graph.show_rag(
        segments, G, image, border_color="yellow", ax=ax, edge_cmap="viridis"
    )
    plt.savefig(save_fold + "-rag.png")
    plt.close()

    fig, ax = plt.subplots(nrows=1, figsize=(18, 16))
    lc = plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_fold + save_name + ".png")
    plt.close()

    add_attr_(image=image, segments=segments, G=G)

    # Visualization
    visualize.show_rag(
        img=image,
        segments=segments,
        g=G,
        save_fold=save_fold,
        attr_color="mean_intensity",
    )
    visualize.show_rag(
        img=image,
        segments=segments,
        g=G,
        save_fold=save_fold,
        attr_color="lzw_complexity",
    )
    # G=nx.read_gml(save_fold+save_name+".gml")
    # segments= np.load('/home/hp/PycharmProjects/graph/superpix/100_segments/JR38_elongation/segments.npy')
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="lzw_complexity",
        save_fold=save_fold,
    )
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="mean_intensity",
        save_fold=save_fold,
    )
    visualize.plot_superpix_image(image, segments, save_fold=save_fold)

    # Saving: first we remove RAG unused features
    rem_attr = ["labels", "pixel count", "total color", "mean color", "centroid"]
    for attr in rem_attr:
        remove_attribute(G, attr)
    # G =rag(image, labels=segments, connectivity=2)
    np.save(save_fold + "segments", segments)
    nx.write_gml(G, save_fold + save_name + ".gml", stringizer=str)
    """
    visualize.degree_analysis(G, save_fold=save_fold, save_name=save_name, title=save_name)
    visualize.adjaciency_matx(G, save_fold=save_fold, save_name=save_name, title=save_name)
    visualize.draw_degree_colormap(G, save_fold=save_fold, save_name=save_name, title=save_name)
    centrality= nx.centrality.information_centrality(G, weight= G.edges(data=True))
    values, bins= np.histogram(list(centrality.values()), 50)
    visualize.histogram(bins, list(centrality.values()), xlabel='Centrality', ylabel='# Nodes',
                        title='Information Centrality '+save_name, save_fold=save_fold, save_name='InfoCentrality_hist_'+save_name)

    """
