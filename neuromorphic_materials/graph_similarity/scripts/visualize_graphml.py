#! /usr/bin/env python3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
)
from neuromorphic_materials.graph_similarity.visualize_graph import visualize_graph


class VisualizeGraphmlParser(Tap):
    image_path: Path = None
    graph_path: Path = None
    only_save_graph: bool = False

    def configure(self) -> None:
        self.add_argument(nameof(self.image_path))
        self.add_argument(nameof(self.graph_path))


def main() -> None:
    args = VisualizeGraphmlParser().parse_args()

    img = cv2.imread(str(args.image_path.resolve()))
    graph: nx.Graph = nx.read_graphml(args.graph_path)

    print(f"number of nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")

    fig, ax = visualize_graph(img, graph)

    if args.only_save_graph:
        fig.savefig(args.graph_path.with_suffix(".png"))
        return

    beta_range = np.logspace(-3, 4, 300)
    se = compute_normalized_spectral_entropy(graph, beta_range)
    fig, ax = plt.subplots()
    ax.plot(beta_range, se)
    ax.set_xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
