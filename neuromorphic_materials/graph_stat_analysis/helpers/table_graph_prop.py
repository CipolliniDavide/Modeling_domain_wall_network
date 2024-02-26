import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain, modularity

from .graph import average_degree, connected_components
from .visual import annotate_heatmap


def table(data, columns, rows_values):
    # data = [[ 66386, 174296,  75131, 577908,  32015],
    #        [ 58230, 381139,  78045,  99308, 160454],
    #        [ 89135,  80552, 152558, 497981, 603535],
    #        [ 78415,  81858, 150656, 193263,  69638],
    #       [139361, 331509, 343164, 781380,  52269]]

    # columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ["%d p" % x for x in rows_values]

    values = np.arange(0, 2500, 500)
    value_increment = 1000

    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(["%1.1f" % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(
        cellText=cell_text,
        rowLabels=rows,
        rowColours=colors,
        colLabels=columns,
        loc="bottom",
    )

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Loss in ${0}'s".format(value_increment))
    plt.yticks(values * value_increment, ["%d" % val for val in values])
    plt.xticks([])
    plt.title("Loss by Disaster")

    plt.show()


def table_imshow(load_path, save_path, title="", p_list_to_plot=None):
    # load_path = '/home/davide/PycharmProjects/graph/superpix-POLAR/JR38_elongation/maxDistdiagonal/'
    # load_path = '/home/davide/PycharmProjects/graph/superpix-POLAR/JR38_elongation/maxDistNone/'
    list_dir = next(os.walk(load_path))[1]

    # Sort by p value
    p_list = []
    for dir in list_dir:
        p_list.append(float(dir.rsplit("p", 1)[1]))
    ind = np.argsort(p_list)
    list_dir = np.asarray(list_dir)[ind]
    p_list = np.asarray(p_list)[ind]

    # index = [np.argwhere(p_list==p) for p in p_list if p in p_list_to_plot]
    # list_dir = list_dir[index]
    # p_list = p_list[index]
    # Gather data
    n_nodes = []
    avg_degree = []
    avg_w_degree = []
    avg_cl = []
    diam = []
    Q = []
    sparsity = []
    for dir in list_dir:
        G = connected_components(
            nx.read_gml(os.path.join(load_path, dir, "graph.gml"))
        )[0]
        n_nodes.append(G.number_of_nodes())
        sp = np.sum(nx.to_numpy_array(G) > 0) / (G.number_of_nodes() ** 2)
        sparsity.append(sp)
        avg_degree.append(average_degree(G))
        # avg_w_degree.append(avg_weighted_degree(G, key_='weight'))
        avg_cl.append(nx.average_clustering(G, weight=None))
        diam.append(nx.diameter(G))
        # partition = [y['partition'] for idx, (x, y) in enumerate(G.nodes(data=True))]
        partition = community_louvain.best_partition(G, weight="weight")
        Q.append(modularity(graph=G, partition=partition, weight="weight"))
        # avg weighted degree
        # a=nx.average_degree_connectivity(G)
    array = np.column_stack(
        (n_nodes, sparsity, avg_degree, avg_cl, diam, Q)
    ).transpose()

    # Plot
    fontdict = {"fontsize": "xx-large", "fontweight": "semibold"}
    fontdict_lab = {"fontsize": 25, "fontweight": "bold"}
    y_ticks = [
        "Number\nof nodes",
        "Sparsity",
        "Avg\ndegree",
        "Avg\nclust.",
        "Diameter",
        "Q",
    ]
    fig, ax = plt.subplots(figsize=(16, 10), nrows=len(array), ncols=1, sharex=True)
    for i, img in enumerate(array):
        im = ax[i].imshow([img], origin="upper")  # , extent=[-2,2,-1,1])
        ax[i].set_yticks([0])
        ax[i].set_yticklabels([y_ticks[i]], fontdict=fontdict_lab)
        annotate_heatmap(
            im=im, valfmt="{x:.2f}", textcolors=("white", "black"), fontdict=fontdict
        )
    y_ticks = ["Avg\ndegree", "Avg\nclust.", "Diameter", "Q"]
    x_ticks = p_list
    ax[i].set_xticks(np.arange(0.05, len(x_ticks) + 0.05))  # [-0.75,-0.25,0.25,0.75])
    ax[i].set_xticklabels(x_ticks, fontdict=fontdict_lab)
    ax[i].set_xlabel(
        "p", fontdict=fontdict_lab
    )  # fontsize='x-large', fontweight='bold')
    plt.suptitle("Largest connected component\n{:s}".format(title), **fontdict_lab)
    plt.tight_layout()
    plt.savefig(save_path + title + "_table_properties.png")
    print("Save table in\n\t{:s}".format(save_path))
