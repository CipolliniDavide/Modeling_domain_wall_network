import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two directories up
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from neuromorphic_materials.graph_scripts.helpers import utils, graph
from neuromorphic_materials.graph_stat_analysis.helpers.graph import merge_nodes_by_distance, connected_components
from pathlib import Path
from neuromorphic_materials.graph_similarity.visualize_graph import visualize_graph

# def visualize_graph(image, graph, alpha=1, node_size=8, edgewidth=2):
#     fig, ax = plt.subplots()
#     ax.imshow(image, interpolation="none", cmap=cm.gray)
#
#     # Draw edges as red lines
#     for edge in graph.edges:
#         start = graph.nodes[edge[0]]
#         end = graph.nodes[edge[1]]
#         ax.plot(
#             [start["x"], end["x"]], [start["y"], end["y"]], color="red", linewidth=edgewidth, alpha=alpha
#         )
#
#     # Make nodes red
#     ax.scatter(
#         [coords["x"] for _, coords in graph.nodes.data()],
#         [coords["y"] for _, coords in graph.nodes.data()],
#         marker="o",
#         s=node_size,
#         c="yellow",
#     )
#
#     ax.axis("off")
#     fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     fig.set_size_inches(5.12, 5.12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp_g', '--load_graph_path', default='../../Dataset/GroundTruth2/graphml/',
                        type=str)
    parser.add_argument('-lp_img', '--load_img_path', default='../../Dataset/GroundTruth2/samples/',
                        type=str)
    parser.add_argument('-eps', '--merge_epsilon', default=1.4, type=float)
    parser.add_argument('-svp', '--save_path', default='../../Dataset/GroundTruth2/annotated_img/',
                        type=str)
    parser.add_argument('-figform', '--fig_format', default='.png', type=str)
    parser.add_argument('-show', '--show_fig', default=False, type=bool)
    parser.add_argument('-n', '--num', default=50, type=int, help='Number of image to produce')
    args = parser.parse_args()



    # root = os.getcwd() #.rsplit('/', 1)[0] +

    #####################################################################

    # args.load_graph_path = os.getcwd()+'/Dataset/Voronoi128_d0.32_p0.20_beta2.95/graphml/'
    # args.load_img_path = os.getcwd()+'/Dataset/Voronoi128_d0.32_p0.20_beta2.95/images/'
    # args.save_path = os.getcwd() + '/Dataset/Voronoi128_d0.32_p0.20_beta2.95/annotated_img2/'
    #
    # args.load_graph_path = os.getcwd() + '/Dataset/GroundTruth2/graphml/'
    # args.load_img_path = os.getcwd() + '/Dataset/GroundTruth2/samples/GridLike'
    # args.save_path = os.getcwd() + '/Dataset/GroundTruth2/annotated_img2/'

    utils.ensure_dir(args.save_path)
    file_list_graph = sorted(glob('{:s}/*.graphml'.format(args.load_graph_path)))
    file_list_img = sorted(glob('{:s}/*.png'.format(args.load_img_path)))

    print(f'Load paths:\n{args.load_graph_path, args.load_img_path}')
    print(f'Number of graphs loaded: {len(file_list_graph)}\nNumber of imgs loaded: {len(file_list_img)}')
    num = args.num
    for i in range(len(file_list_graph[:num])):
        img = plt.imread(file_list_img[i])
        G = nx.read_graphml(file_list_graph[i])
        G, list_of_rem_nodes = merge_nodes_by_distance(g=G, epsilon=args.merge_epsilon, return_removed_nodes=True)
        G = connected_components(G)[0]
        visualize_graph(image=img, graph=G, node_size=80, c_node='red', c_edge='yellow', alpha_edge=.5)
        name = file_list_graph[i].rsplit('/', 1)[1].rsplit('.', 1)[0]
        plt.tight_layout()
        plt.savefig(args.save_path+name+args.fig_format, dpi=300)
        plt.close()

    print('{:d} figures saved to:\n\t{:s}'.format(len(file_list_graph[:num]), args.save_path))

