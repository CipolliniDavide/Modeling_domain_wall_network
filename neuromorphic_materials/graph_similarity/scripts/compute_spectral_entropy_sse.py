#! /usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
    compute_normalized_spectral_entropy_sse,
)


class ComputeSpectralEntropySSEParser(Tap):
    # Assign None to ensure nameof works in configure
    graph_one_path: Path = None
    graph_two_path: Path = None

    def configure(self) -> None:
        self.add_argument(nameof(self.graph_one_path))
        self.add_argument(nameof(self.graph_two_path))


def main() -> None:
    args = ComputeSpectralEntropySSEParser().parse_args()

    graph_one = nx.read_graphml(args.graph_one_path)
    graph_two = nx.read_graphml(args.graph_two_path)
    beta_range = np.logspace(-3, 4, 300)

    sse = compute_normalized_spectral_entropy_sse(graph_one, graph_two, beta_range)
    print(f"SE SSE: {sse}")
    se_one = compute_normalized_spectral_entropy(graph_one, beta_range)
    se_two = compute_normalized_spectral_entropy(graph_two, beta_range)

    plt.plot(beta_range, se_one, label="BFO")
    plt.plot(beta_range, se_two, label="Voronoi")
    plt.legend(loc="upper right")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
