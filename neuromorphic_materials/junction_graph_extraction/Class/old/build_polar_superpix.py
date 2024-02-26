import argparse
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from matplotlib import cm
from skimage import io
from skimage.color import gray2rgb
from skimage.future import graph
from skimage.segmentation import slic
from sklearn.manifold import TSNE

sys.path.extend(["/home/davide/PycharmProjects/graph/scripts"])
from Class.load_graph import connected_components
from Class.superpix_graph import add_attr_, rag, remove_attribute, superpix_graph
from Class.table_graph_prop import table_imshow
from Class.utils import utils
from Class.visualize import visualize

# save_name = 'JR38_elongation'
# roothpath = '/home/davide/PycharmProjects/graph/'


def make_smart(image, segments, args):
    # Create graph
    super_pix = superpix_graph(
        image=image,
        segments=segments,
        save_fold=save_fold,
        max_distance=args.max_distance,
    ).create_graph()

    fig, ax = plt.subplots(nrows=1, figsize=(18, 16))
    lc = plt.imshow(image, "gray", origin="lower")
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_fold + args.save_name + ".png")
    plt.close()

    # Saving: first we remove RAG unused features
    # rem_attr = ['labels', 'pixel count', 'total color', 'mean color', 'centroid']
    # for attr in rem_attr: remove_attribute(G, attr)
    np.save(save_fold + "segments", segments)


def make_(image, segments, args):
    # Create graph
    G = superpix_graph(
        p=p,
        image=image,
        segments=segments,
        save_fold=save_fold,
        max_distance=args.max_distance,
    ).create_graph()

    fig, ax = plt.subplots(nrows=1, figsize=(18, 16))
    lc = plt.imshow(image, "gray", origin="lower")
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_fold + args.save_name + ".png")
    plt.close()

    # Saving: first we remove RAG unused features
    # rem_attr = ['labels', 'pixel count', 'total color', 'mean color', 'centroid']
    # for attr in rem_attr: remove_attribute(G, attr)
    np.save(save_fold + "segments", segments)
    nx.write_gml(G, save_fold + "graph" + ".gml", stringizer=str)
    return G, segments


def centrality_plots(image, segments, G, save_fold, save_name):
    # centrality= nx.get_node_attributes(G, 'information_centrality')
    centrality = nx.centrality.information_centrality(G, weight=G.edges(data=True))
    _, bins = np.histogram(list(centrality.values()), 50)
    visualize.histogram(
        bins,
        list(centrality.values()),
        xlabel="Information Centrality",
        ylabel="# Nodes",
        title="Information Centrality " + save_name,
        save_fold=save_fold,
        save_name="InfoCentrality_hist_" + save_name,
    )
    nx.set_node_attributes(G, centrality, "information_centrality")
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="information_centrality",
        save_fold=save_fold,
    )

    betweenness = nx.betweenness_centrality(G)
    # betweenness = nx.get_node_attributes(G,'betweenness_centrality')
    _, bins = np.histogram(list(betweenness.values()), 50)
    visualize.histogram(
        bins,
        list(betweenness.values()),
        xlabel="Betweenness Centrality",
        ylabel="# Nodes",
        title="Betweenness Centrality " + save_name,
        save_fold=save_fold,
        save_name="betweenness_centrality" + save_name,
    )
    nx.set_node_attributes(G, centrality, "betweenness_centrality")
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="betweenness_centrality",
        save_fold=save_fold,
    )


def standard_analysis(image, segments, G, save_fold, save_name):
    visualize.degree_analysis(
        G, save_fold=save_fold, save_name=save_name, title=save_name
    )
    # visualize.draw_degree_colormap(G, save_fold=save_fold, save_name=save_name, title=save_name)
    visualize.adjaciency_matx(
        G, save_fold=save_fold, save_name=save_name, title=save_name
    )
    visualize.plot_graph_over_image(G, segments, image, save_fold=save_fold)
    visualize.plot_superpix_image_point_color(
        image=image, segments=segments, G=G, attr_color="lzw", save_fold=save_fold
    )
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="mean_intensity",
        save_fold=save_fold,
    )
    # visualize.plot_superpix_image_point_color(image=image, segments=segments, G=G, attr_color='information_centrality', save_fold=save_fold)
    # visualize.plot_superpix_image_point_color(image=image, segments=segments, G=G, attr_color='betweenness_centrality', save_fold=save_fold)

    weight = list(nx.get_edge_attributes(G, "weight").values())
    visualize.histogram(
        bins=np.histogram(weight, 30)[1],
        values=weight,
        xlabel="Weight",
        ylabel="#",
        title="Weights Distribution " + save_name,
        save_fold=save_fold,
        save_name="weights_" + save_name,
    )


def community_detection(G):
    # compute the best partition
    resolution = 1.0
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution
    )
    nx.set_node_attributes(G, partition, "partition")
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=G,
        attr_color="partition",
        cbar_type="discrete",
        save_fold=save_fold,
    )
    return G


# for numSegments in [150]:#, 300, 500]:#, 150, 200]:#, 100, 150]:#10, 100, 200, 300, 500]:
if __name__ == "__main__":
    # define parser & its arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rooth_path",
        "--rooth_path",
        default="/home/davide/PycharmProjects/graph/",
        type=str,
    )
    parser.add_argument(
        "-save_name", "--save_name", default="JR38_elongation", type=str
    )
    parser.add_argument(
        "-nSeg", "--numSegments", default=150, help="Number of Superpixels", type=int
    )
    parser.add_argument(
        "-sigma", "--sigma", default=2, help="Sigma for slic", type=float
    )
    parser.add_argument("-s", "--show", default=False, nargs="?", help="Show Plot")
    parser.add_argument(
        "-p", "--p", default=0.0001, help="Probability to erase weights", type=float
    )
    parser.add_argument(
        "-max_d",
        "--max_distance",
        default="diagonal",
        help=(
            "Connect only nodes closer than max_distance. Options are: 'diagonal',"
            " 'None'."
        ),
        type=str,
    )
    parser.add_argument(
        "-load",
        "--load_path",
        nargs=1,
        default=(
            "/home/davide/PycharmProjects/graph/JR38_CoCr_6_70000.ibw_4_12_2_3.pgm_.png"
        ),
        type=str,
    )
    # parse
    args = parser.parse_args()

    # args.sigma = sigma
    # load image
    image = io.imread(args.load_path)
    # apply SLIC and extract (approximately) the supplied number of segments
    image_rgb = gray2rgb(image)
    # Build graph with SLIC
    segments = slic(
        image_rgb,
        n_segments=args.numSegments,
        sigma=args.sigma,
        start_label=1,
        convert2lab=True,
    )
    # segments = slic(image, compactness=30, n_segments=numSegments, start_label=1)

    args.rooth_path = (
        args.rooth_path
        + "superpix-POLAR_test/{}/maxDist{}/{}segments_sigma{}/".format(
            args.save_name, args.max_distance, args.numSegments, args.sigma
        )
    )

    # for p in [0., .2, .3, .4, .5, .6, .7]:
    for p in [0.6, 0.7, 0.8]:  # ,.2]:#, 0.05]:#, .1, .2, .3, .4, .5]:
        args.p = p
        save_fold = args.rooth_path + "p{:.3f}/".format(args.p)
        # save_fold = '/home/davide/PycharmProjects/ESN/load_graph_folder_test/'
        utils.ensure_dir(save_fold)
        print("Save to {}".format(save_fold))
        # Save args on txt
        with open(save_fold + "args.txt", "w") as fp:
            fp.write(str(args))

        visualize.plot_superpix_image(image, segments, save_fold=save_fold)
        # Make superpix graph
        # se tirassi fuori il pruning sarebbe meglio
        G, segments = make_(image=image, segments=segments, args=args)

        # Select largest connected component
        # G = nx.read_gml('/home/davide/PycharmProjects/graph/superpix-POLAR_test/150segments_sigma1.5_p0.01_maxDistNone/JR38_elongation/JR38_elongation.gml')
        # segments = np.load('/home/davide/PycharmProjects/graph/superpix-POLAR_test/150segments_sigma1.5_p0.01_maxDistNone/JR38_elongation/segments.npy')

        G = connected_components(G)[0]

        # Community detection
        G = community_detection(G)

        # Analysis of largest connected component
        standard_analysis(
            image=image,
            segments=segments,
            G=G,
            save_fold=save_fold,
            save_name=args.save_name,
        )

        # Centrality
        centrality_plots(
            image=image,
            segments=segments,
            G=G,
            save_fold=save_fold,
            save_name=args.save_name,
        )

        # Save
        nx.write_gml(G, save_fold + "graph.gml", stringizer=str)

    table_imshow(load_path=args.rooth_path)

    # draw the graph
    # pos = nx.spring_layout(G)
    # color the nodes according to their partition
    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    #                       cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()
