#! /usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_log_likelihood,
)


class ComputeLogLikelihoodParser(Tap):
    # Assign None to ensure nameof works in configure
    graph_one_path: Path = None
    graph_two_path: Path = None

    def configure(self) -> None:
        self.add_argument(nameof(self.graph_one_path))
        self.add_argument(nameof(self.graph_two_path))


def main() -> None:
    args = ComputeLogLikelihoodParser().parse_args()

    graph_one = nx.read_graphml(args.graph_one_path)
    graph_two = nx.read_graphml(args.graph_two_path)
    beta_range = np.logspace(-3, 4, 300)

    re = compute_log_likelihood(
        nx.laplacian_matrix(graph_one).toarray(),
        nx.laplacian_matrix(graph_two).toarray(),
        beta_range,
    )

    plt.plot(beta_range, re, label="Log likelihood")
    plt.legend(loc="lower right")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
