#! /usr/bin/env python3
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
)
from neuromorphic_materials.graph_similarity.visualize_graph import visualize_graph
from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.sample_generation.src.voronoi.arg_parser import (
    VoronoiParser,
)
from neuromorphic_materials.sample_generation.src.voronoi.generator import (
    VoronoiGenerator,
)


def main() -> None:
    args = VoronoiParser().parse_args()
    beta_range = np.logspace(-3, 4, 300)

    generator = VoronoiGenerator.from_arg_parser_args(args)

    n_cvds = 16

    voronoi_samples = generator.generate_ensemble_boundaries(n_cvds)

    fig, ax = plt.subplots(figsize=(8, 6))

    mean_ensemble_se = np.zeros_like(beta_range)
    for idx, sample in enumerate(voronoi_samples):
        graph = extract_graph_from_binary_voronoi(
            sample, Path("../../junction_graph_extraction/beyondOCR_junclets/my_junc")
        )

        se = compute_normalized_spectral_entropy(graph, beta_range)
        visualize_graph(sample, graph)
        mean_ensemble_se += se
        ax.plot(beta_range, se, label=f"Sample {idx + 1}", linestyle="dotted")

    mean_ensemble_se /= n_cvds
    ax.plot(beta_range, mean_ensemble_se, label="Average", linestyle="dashed")

    ax.legend(loc="lower right")
    ax.set_ylabel(
        "Normalized Von Neumann entropy\n" r"$S(\mathbf{\rho}_{\beta})\//\/\log(N)$"
    )
    ax.set_xlabel(r"$\beta^{-1}$")
    ax.set_xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
