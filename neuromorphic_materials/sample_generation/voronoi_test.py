#! /usr/bin/env python3
import os
import sys
sys.path.insert(1, '{:s}/'.format(os.getcwd()))
import networkx as nx
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.pdf import uniform_point
from src.point import Point
from src.voronoi.arg_parser import VoronoiParser
from src.voronoi.generator import VoronoiGenerator
# from src.voronoi.generator_gpu import VoronoiGeneratorGPU
from src.voronoi.site import VoronoiSite

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
)
from neuromorphic_materials.graph_similarity.visualize_graph import visualize_graph
from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.graph_stat_analysis.helpers.graph import merge_nodes_by_distance, connected_components
from neuromorphic_materials.graph_scripts.helpers import utils

def show_plot() -> None:
    plt.axis("off")
    plt.tight_layout()
    plt.margins(0, 0)
    plt.show()


def compute_voronoi_naive(sample_size: int, sites: list[VoronoiSite]) -> np.ndarray:
    sample = np.empty((sample_size, sample_size), np.int_)
    for x in range(sample_size):
        for y in range(sample_size):
            sample[x, y] = sites.index(
                min(sites, key=lambda site: site.dist_to_point(Point(x, y)))
            )
    return sample


def plot_voronoi(voronoi_sample: np.ndarray, sites: list[VoronoiSite], show=True) -> None:
    fig = plt.figure(figsize=(5,5))
    plt.imshow(voronoi_sample, interpolation="none", cmap="coolwarm")
    plt.axis('off')
    for site in sites:
        plt.scatter(site.y, site.x, c="black", marker=".")
    if show:
        show_plot()
    plt.tight_layout()


def main() -> None:
    parser = VoronoiParser()
    args = parser.parse_args()

    tick = time.perf_counter()
    generator = VoronoiGenerator(
        args.site_linear_density,
        args.p_horizontal,
        args.horizontal_beta,
        args.sample_size,
        args.n_iterations,
        args.site_update_method,
        uniform_point(0, args.sample_size, np.random.default_rng()),
        # np.random.default_rng(args.rng_seed),
    )

    utils.ensure_dir(str(args.save_binary)+'/')
    binary_img_path = str(args.save_binary) + '/' + 'binary/'
    voronoi_save_path = str(args.save_binary) + '/' + 'voronoi/'
    spec_entropy_save_path = str(args.save_binary) + '/' + 'spec-entropy/'
    annotated_img_save_path = str(args.save_binary) + '/' + 'annotated/'
    graph_img_save_path = str(args.save_binary) + '/' + 'graph_img/'
    for s in [binary_img_path, voronoi_save_path, spec_entropy_save_path, annotated_img_save_path, graph_img_save_path]:
        utils.ensure_dir(s)

    print(f"Instantiated VoronoiGenerator in {time.perf_counter() - tick:0.4f} seconds")
    print(args)
    tick = time.perf_counter()
    voronoi_samples = generator.generate_cvds(10)

    for i in range(len(voronoi_samples)):
        voronoi_sample = voronoi_samples[i]
        print(f"Computed CVT in {time.perf_counter() - tick:0.4f} seconds")
        print('Iterations', args.n_iterations)
        plot_voronoi(voronoi_sample.diagram, voronoi_sample.sites, show=False)
        plt.savefig(str(voronoi_save_path) + '/{:04d}voronoi.jpg'.format(i), dpi=300)
        plt.close()

        fig = plt.figure(figsize=(5,5))
        plt.gray()
        plt.imshow(voronoi_sample.boundaries, interpolation="none")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(binary_img_path)+'/{:04d}boundaries.jpg'.format(i), dpi=300)
        plt.close()
        # show_plot()

        tick = time.perf_counter()
        graph = extract_graph_from_binary_voronoi(
            voronoi_sample.boundaries,
            Path("../junction_graph_extraction/beyondOCR_junclets/my_junc"),
        )
        graph, rem_nodes = merge_nodes_by_distance(g=graph, epsilon=1.4, return_removed_nodes=True)
        graph = connected_components(graph)[0]
        print(f"Extracted graph in {time.perf_counter() - tick:0.4f} seconds")

        G = graph.copy()
        G.remove_edges_from(nx.selfloop_edges(graph))
        graph = G

        fig, ax = plt.subplots()
        nx.draw_networkx(graph, node_size=80, node_shape='o', node_color='red', with_labels=False,
                         pos=nx.spring_layout((graph)),
                         ax=ax, edge_color='black',
                         width=5, linewidths=4)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(graph_img_save_path+'{:04d}graph.svg'.format(i))
        plt.close()

        tick = time.perf_counter()
        beta_range = np.logspace(-3, 4, 300)
        voronoi_se = compute_normalized_spectral_entropy(graph, beta_range)
        print(f"Computed SE in {time.perf_counter() - tick:0.4f} seconds")
        plt.plot(beta_range, voronoi_se)
        plt.xscale("log")
        plt.savefig(str(spec_entropy_save_path)+'{:04d}specent.jpg'.format(i), dpi=300)
        # plt.show()
        plt.close()

        print(f"number of nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")

        visualize_graph(voronoi_sample.boundaries, graph, node_size=80, c_node='red', c_edge='yellow', alpha_edge=.5)
        plt.savefig(str(annotated_img_save_path) + '/{:04d}annotated_img.jpg'.format(i), dpi=300)
        # plt.show()
        plt.close()

        if args.save_binary is not None:
            # save_path = args.save_binary / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            # print(f"Saving binary image to\n{save_path.resolve()}")
            # cv2.imwrite(str(save_path), voronoi_sample.boundaries)
            cv2.imwrite(str(args.save_binary) + '/{:04d}binary.jpg'.format(i), voronoi_sample.boundaries)


if __name__ == "__main__":
    main()
