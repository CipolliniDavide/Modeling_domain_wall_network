#! /usr/bin/env python3
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
)
from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.sample_generation.src.pdf import uniform_point
from neuromorphic_materials.sample_generation.src.voronoi.arg_parser import (
    VoronoiParser,
)
from neuromorphic_materials.sample_generation.src.voronoi.generator import (
    VoronoiGenerationError,
    VoronoiGenerator,
)


def main() -> None:
    args = VoronoiParser().parse_args()
    generator = VoronoiGenerator(
        args.site_linear_density,
        args.p_horizontal,
        args.horizontal_beta,
        args.sample_size,
        args.n_iterations,
        "centroid",
        uniform_point(0, args.sample_size, np.random.default_rng()),
    )

    beta_range = np.logspace(-3, 4, 300)
    colors = ["red", "blue", "green", "purple", "cyan"]

    for _ in range(1000):
        try:
            sample = generator.generate_ensemble_boundaries(1)
        except VoronoiGenerationError as e:
            print(e)
            continue
        graph = extract_graph_from_binary_voronoi(
            sample, Path("../../junction_graph_extraction/beyondOCR_junclets/my_junc")
        )
        graph.remove_nodes_from([node for node, deg in graph.degree if deg == 0])
        connected_components: list[nx.Graph] = [
            graph.subgraph(cc) for cc in nx.connected_components(graph)
        ]
        if len(connected_components) > 1:
            print(f"Number of CCs: {len(list(connected_components))}")
            se = compute_normalized_spectral_entropy(graph, beta_range)
            plt.plot(beta_range, se)
            plt.xscale("log")
            plt.show()
            print(f"Max SE: {se.max()}")
            plt.imshow(sample.boundaries, interpolation="none")

            for subgraph, color in zip(connected_components, colors):
                print(f"CC size: {subgraph.number_of_nodes()}")
                plt.scatter(
                    [coords["x"] for _, coords in subgraph.nodes.data()],
                    [coords["y"] for _, coords in subgraph.nodes.data()],
                    marker=".",
                    s=5,
                    c=color,
                )

                # Draw edges as red lines
                for edge in subgraph.edges:
                    start = subgraph.nodes[edge[0]]
                    end = subgraph.nodes[edge[1]]
                    plt.plot(
                        [start["x"], end["x"]],
                        [start["y"], end["y"]],
                        color=color,
                        linewidth=1,
                    )
            plt.show()
            break


if __name__ == "__main__":
    main()
