import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# def visualize_graph(image: np.ndarray, graph: nx.Graph):
#     fig, ax = plt.subplots()
#     ax.imshow(image, cmap="gray", interpolation="none")
#
#     # Make nodes red
#     ax.scatter(
#         [coords["x"] for _, coords in graph.nodes.data()],
#         [coords["y"] for _, coords in graph.nodes.data()],
#         marker=".",
#         s=5,
#         c="red",
#     )
#
#     # Draw edges as red lines
#     for edge in graph.edges:
#         start = graph.nodes[edge[0]]
#         end = graph.nodes[edge[1]]
#         ax.plot(
#             [start["x"], end["x"]], [start["y"], end["y"]], color="red", linewidth=1
#         )
#
#     ax.axis("off")
#     fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     fig.set_size_inches(5.12, 5.12)
#
#     return fig, ax

import matplotlib.cm as cm
def visualize_graph(image: np.ndarray,
                    graph: nx.Graph,
                    alpha_edge: float=.35,
                    edgewidth: float=3.5,
                    node_size: float=50,
                    c_node: str='yellow',
                    c_edge: str='red'):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(image, interpolation="none", cmap=cm.gray)

    # Draw edges as red lines
    for edge in graph.edges:
        start = graph.nodes[edge[0]]
        end = graph.nodes[edge[1]]
        ax.plot(
            [start["x"], end["x"]], [start["y"], end["y"]], color=c_edge, linewidth=edgewidth, alpha=alpha_edge
        )

    # Make nodes red
    ax.scatter(
        [coords["x"] for _, coords in graph.nodes.data()],
        [coords["y"] for _, coords in graph.nodes.data()],
        marker="o",
        s=node_size,
        alpha=1,
        c=c_node,
    )

    ax.axis("off")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.set_size_inches(5.12, 5.12)
    plt.tight_layout()
    return fig, ax