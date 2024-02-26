#! /usr/bin/env python3
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import Graph
from src.delaunay.arg_parser import DelaunaySampleParser
from src.delaunay.generator import DelaunaySampleGenerator


def _degree_analysis(graph: Graph) -> None:
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    fig = plt.figure("Degree analysis of sample graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    graph_cc = graph.subgraph(
        sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    )
    pos = nx.get_node_attributes(graph_cc, "coords")
    nx.draw_networkx_nodes(graph_cc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(graph_cc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components")
    ax0.set_axis_off()
    ax0.invert_yaxis()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()


def main() -> None:
    # Default args
    args = DelaunaySampleParser().parse_args(["data/delaunay/test_graph"])

    sample = DelaunaySampleGenerator(args).generate_sample()
    sample_graph = sample.generate_graph()
    print(f"Sample nodes: {sample.n_nodes}, edges: {sample.n_edges}")
    sample.generate_image().save(args.save_dir / "sample.png")

    _degree_analysis(sample_graph)


if __name__ == "__main__":
    main()
