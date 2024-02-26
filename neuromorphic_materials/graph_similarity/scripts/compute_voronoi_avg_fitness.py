#! /usr/bin/env python3
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_ensemble_se,
    compute_normalized_spectral_entropy,
    compute_spectral_entropy,
)
from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.sample_generation.src.voronoi.generator import (
    VoronoiGenerator,
)


def compute_bfo_ensemble_se(beta_range, normalized) -> np.ndarray:
    bfo_graph_count = 0
    bfo_ensemble_se = np.zeros_like(beta_range)
    compute_se_method = (
        compute_normalized_spectral_entropy if normalized else compute_spectral_entropy
    )
    for graph_file in Path("../../../data/graph_annotation").glob("*.graphml"):
        bfo_graph_count += 1
        se = compute_se_method(nx.read_graphml(graph_file), beta_range)
        bfo_ensemble_se += se

    bfo_ensemble_se /= bfo_graph_count
    return bfo_ensemble_se


def main() -> None:
    beta_range = np.logspace(-3, 4, 300)

    n_ensembles = 100
    ensemble_size = 16
    normalized = False
    sample_size = 128
    n_iterations = 10
    # site_linear_density = 0.4724
    # p_horizontal = 0.4065
    # horizontal_beta = 2.9207
    site_linear_density = 0.4849
    p_horizontal = 0.4130
    horizontal_beta = 2.8790
    junclets_path = Path("../../junction_graph_extraction/beyondOCR_junclets/my_junc")

    bfo_ensemble_se = compute_bfo_ensemble_se(beta_range, normalized)

    fitnesses = np.empty(n_ensembles)
    generator = VoronoiGenerator(
        site_linear_density,
        p_horizontal,
        horizontal_beta,
        sample_size,
        n_iterations,
        "centroid",
    )
    for idx in tqdm(range(n_ensembles)):
        samples = generator.generate_ensemble_boundaries_no_except(ensemble_size)
        voronoi_ensemble_mean_se = compute_ensemble_se(
            [
                extract_graph_from_binary_voronoi(sample, junclets_path)
                for sample in samples
            ],
            beta_range,
            normalized,
        )

        fitnesses[idx] = 1 / ((bfo_ensemble_se - voronoi_ensemble_mean_se) ** 2).sum()

    print(
        f"Max: {fitnesses.max()}, min: {fitnesses.min()} mean: {fitnesses.mean()},"
        f" stddev: {fitnesses.std()}"
    )
    save_path = Path(
        f"../../../data/voronoi_avg_fitness/{n_ensembles}_ensembles_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, fitnesses)


if __name__ == "__main__":
    main()
