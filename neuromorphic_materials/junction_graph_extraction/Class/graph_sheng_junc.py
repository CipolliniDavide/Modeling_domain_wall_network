import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from . import nefi_short


def graph_sheng_junc(features, coordinates, image, binary_img, save_fold="./"):
    G_sheng = nx.Graph()
    node_list = [
        ((coord[1], coord[0]), {"x": feat})
        for coord, feat in zip(coordinates, features)
    ]
    G_sheng.add_nodes_from(
        node_list
    )  # cx= cx, cy= cy, mean_intensity= mean_a, lzw_complexity=lzw)
    # otsued = binary_img  # nefi_short.otsu_process([image])['img']
    otsued = nefi_short.otsu_process([image])["img"]
    plt.imshow(otsued)
    plt.savefig(save_fold + "otsu.png")
    plt.close()
    otsued = binary_img  # otsued//255
    skeleton = nefi_short.thinning([otsued])["skeleton"]
    G_sheng = nefi_short.breadth_first_edge_detection(skeleton, otsued, G_sheng)
    # coord_sheng= np.asarray([ [y, x] for x, y in G.nodes() ])
    return G_sheng


def plot_graph(features, coordinates, G_sheng, image, save_fold="./"):
    # Plot graph on image
    fig = plt.figure()  # figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, "gray")
    ax.scatter(coordinates[:, 0], coordinates[:, 1], marker="+", c="yellow")
    for n1, n2, attr in G_sheng.edges(data=True):
        l = Line2D(
            [n2[1], n1[1]], [n2[0], n1[0]], alpha=0.4, linewidth=attr["width"]
        )  # c=attr['pixels'],
        ax.add_line(l)
    # plt.scatter(coordinates_refined[:,0], coordinates_refined[:,1], marker="+", c='yellow')
    plt.savefig(save_fold + "sheng_detect.png")
    plt.close()


def plot_graph_node_polarplot(features, coordinates, G_sheng, image, save_fold="./"):
    # Plot polar plots
    from .polar_plot import polar_plot
    from .visual_utils import plot_axes

    fontdict = {"font_size_xlabel": 20, "font_size_ylabel": 15, "fontweight": "bold"}
    # x= nx.get_node_attributes(G_sheng, 'x')
    for i in range(len(features)):
        # if i==7: break
        fig = plt.figure()  # figsize=(16, 12))
        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(image, "gray")
        ax.scatter(coordinates[:, 0], coordinates[:, 1], marker="+", c="red")
        ax.set_xticks([]), ax.set_yticks([])
        for n1, n2, attr in G_sheng.edges(data=True):
            # print(attr)
            l = Line2D(
                [n2[1], n1[1]], [n2[0], n1[0]], alpha=0.4, linewidth=attr["width"]
            )  # c=attr['pixels'],
            ax.add_line(l)
        node_x, node_y = coordinates[i, 0], coordinates[i, 1]
        ax.scatter(
            node_x,
            node_y,
            marker="o",
            c="none",
            edgecolor="red",
            facecolors="None",
            s=150,
        )
        # circle = plt.Circle((node_x, node_y), 8, color='yellow')
        # ax.add_patch(circle)
        data = features[i]
        # fig = plot_axes(ax=ax, fig=fig, geometry=(2, 1, 1))
        polar_plot(
            data,
            fontdict=fontdict,
            fig=fig,
            geometry=(212),
            figsize=(24, 16),
            save_fold=save_fold,
            save_name="sheng{:02d}".format(i),
        )
