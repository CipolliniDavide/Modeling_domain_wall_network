#! /usr/bin/env python3
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
    compute_spectral_entropy,
    compute_voronoi_ensemble_mean_se,
)


def main() -> None:
    beta_range = np.logspace(-3, 4, 300)
    site_linear_density = 0.4724
    p_horizontal = 0.4065
    horizontal_beta = 2.9207
    normalized = False
    print(
        "Computing Voronoi Ensemble Mean SE with"
        f" params:\n{site_linear_density=}\n{p_horizontal=}\n{horizontal_beta=}\n"
    )
    (
        voronoi_ensemble_mean_se,
        voronoi_avg_node_count,
        voronoi_avg_edge_count,
    ) = compute_voronoi_ensemble_mean_se(
        beta_range,
        site_linear_density,
        p_horizontal,
        horizontal_beta,
        128,
        10,
        16,
        Path("../../junction_graph_extraction/beyondOCR_junclets/my_junc"),
        normalized,
    )

    bfo_graph_count = 0
    bfo_avg_node_count = 0
    bfo_avg_edge_count = 0
    bfo_ensemble_mean_se = np.zeros_like(beta_range)
    compute_se_method = (
        compute_normalized_spectral_entropy if normalized else compute_spectral_entropy
    )
    for graph_file in Path("../../../data/graph_annotation").glob("*.graphml"):
        bfo_graph_count += 1
        graph = nx.read_graphml(graph_file)
        bfo_avg_node_count += graph.number_of_nodes()
        bfo_avg_edge_count += graph.number_of_edges()
        se = compute_se_method(nx.read_graphml(graph_file), beta_range)
        bfo_ensemble_mean_se += se

    bfo_ensemble_mean_se /= bfo_graph_count
    bfo_avg_node_count /= bfo_graph_count
    bfo_avg_edge_count /= bfo_graph_count

    print(
        "Average node and edge count\n"
        f"  BFO: {bfo_avg_node_count:0.2f}, {bfo_avg_edge_count:0.2f}\n"
        f"  Voronoi: {voronoi_avg_node_count:0.2f}, {voronoi_avg_edge_count:0.2f}"
    )

    sse = ((bfo_ensemble_mean_se - voronoi_ensemble_mean_se) ** 2).sum()
    print(f"SSE: {sse:0.4f}, fitness: {1 / sse: 0.4f}")

    # deriv_bfo = np.gradient(bfo_ensemble_mean_se)
    # deriv_voronoi = np.gradient(voronoi_ensemble_mean_se)
    # deriv_sse = ((deriv_bfo - deriv_voronoi) ** 2).sum()
    # print(f"deriv SSE: {deriv_sse:0.4f}, deriv fitness: {1 / deriv_sse: 0.4f}")

    # _, sse_axes = plt.subplots()
    # sse_axes.plot(beta_range, voronoi_ensemble_mean_se, label="Voronoi")
    # sse_axes.plot(beta_range, bfo_ensemble_mean_se, label="BFO")
    # sse_axes.set_title(
    #     f"Site dens. {site_linear_density:0.4f}, Horiz. prob. {p_horizontal:0.4f},"
    #     f" Chebyshev $\\beta$ {horizontal_beta:0.4f}"
    # )
    # sse_axes.legend(loc="lower right")
    # sse_axes.set_ylabel("Von Neumann entropy\n" r"$S(\mathbf{\rho}_{\beta})$")
    # sse_axes.set_xlabel(r"$\beta$")
    # sse_axes.set_xscale("log")

    # _, sse_deriv_axes = plt.subplots()
    # sse_deriv_axes.plot(beta_range, deriv_voronoi, label="Voronoi")
    # sse_deriv_axes.plot(beta_range, deriv_bfo, label="BFO")
    # sse_deriv_axes.set_title(
    #     f"Site dens. {site_linear_density:0.4f}, Horiz. prob. {p_horizontal:0.4f},"
    #     f" Chebyshev $\\beta$ {horizontal_beta:0.4f}"
    # )
    # sse_deriv_axes.legend(loc="lower right")
    # sse_deriv_axes.set_ylabel(
    #     "Derivative of Von Neumann entropy\n"
    #     r"$\frac{ \mathrm{d} S(\mathbf{\rho}_{\beta})}{ \mathrm{d}\beta }$"
    # )
    # sse_deriv_axes.set_xlabel(r"$\beta$")
    # sse_deriv_axes.set_xscale("log")

    # plt.show()


if __name__ == "__main__":
    main()
