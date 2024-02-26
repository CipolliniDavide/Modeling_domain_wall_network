from pathlib import Path

import networkqit as nq
import networkx as nx
import numpy as np

from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.sample_generation.src.pdf import uniform_point
from neuromorphic_materials.sample_generation.src.voronoi.generator import (
    VoronoiGenerator,
)


def laplacian_renormalization_group(eigenvalues: np.ndarray, new_size: int):
    assert (
        eigenvalues.size > new_size
    ), "New size must be smaller than number of eigenvalues"
    eigenvalues.sort()
    reduced_eigenvalues = eigenvalues[:new_size]
    return reduced_eigenvalues / reduced_eigenvalues[-1]


def compute_log_likelihood(
    l_observed: np.ndarray, l_model: np.ndarray, beta_range: np.ndarray
) -> np.ndarray:
    eigvals_obs = np.linalg.eigvalsh(l_observed)
    eigvals_model = np.linalg.eigvalsh(l_model)

    # Ensuring equal sized graphs
    n_eigvals_obs = len(eigvals_obs)
    n_eigvals_model = len(eigvals_model)
    if n_eigvals_obs > n_eigvals_model:
        eigvals_obs = laplacian_renormalization_group(eigvals_obs, n_eigvals_model)
    elif n_eigvals_model > n_eigvals_obs:
        eigvals_model = laplacian_renormalization_group(eigvals_model, n_eigvals_obs)

    # Produces (n_eigenvalues, n_betas)
    obs_exp = np.exp(-beta_range * eigvals_obs[:, np.newaxis])
    # log L(density_model) = - Tr[rho * L_rand] - log Tr[exp(-beta * L_rand)]
    return -((obs_exp / obs_exp.sum(axis=0)).T * eigvals_model).sum(axis=1) - np.log(
        np.exp(-beta_range * eigvals_model[:, np.newaxis]).sum(axis=0)
    )


def compute_normalized_spectral_entropy(
    graph: nx.Graph, beta_range: np.ndarray
) -> np.ndarray:
    return 1 - (
        nq.entropy(nx.laplacian_matrix(graph).toarray(), beta_range)
        / np.log(graph.number_of_nodes())
    )


def compute_spectral_entropy(graph: nx.Graph, beta_range: np.ndarray) -> np.ndarray:
    return nq.entropy(nx.laplacian_matrix(graph).toarray(), beta_range)


def compute_spectral_entropy_sse(
    bfo_graph: nx.Graph, voronoi_graph: nx.Graph, beta_range: np.ndarray
) -> np.ndarray:
    return (
        (
            compute_spectral_entropy(bfo_graph, beta_range)
            - compute_spectral_entropy(voronoi_graph, beta_range)
        )
        ** 2
    ).sum()


def compute_normalized_spectral_entropy_sse(
    bfo_graph: nx.Graph, voronoi_graph: nx.Graph, beta_range: np.ndarray
) -> np.ndarray:
    return (
        (
            compute_normalized_spectral_entropy(bfo_graph, beta_range)
            - compute_normalized_spectral_entropy(voronoi_graph, beta_range)
        )
        ** 2
    ).sum()


def compute_ensemble_se(
    graphs: list[nx.Graph], beta_range: np.ndarray, normalized: bool
) -> np.ndarray:
    ensemble_se = np.zeros_like(beta_range)
    compute_se_method = (
        compute_normalized_spectral_entropy if normalized else compute_spectral_entropy
    )
    for graph in graphs:
        ensemble_se += compute_se_method(graph, beta_range)

    return ensemble_se / len(graphs)


def compute_voronoi_ensemble_mean_se(
    beta_range: np.ndarray,
    site_linear_density: float,
    p_horizontal: float,
    horizontal_beta: float,
    sample_size: int,
    n_iterations: int,
    batch_size: int,
    junclets_executable_path: Path,
    normalized: bool = False,
) -> np.ndarray:
    generator = VoronoiGenerator(
        site_linear_density,
        p_horizontal,
        horizontal_beta,
        sample_size,
        n_iterations,
        "centroid",
        uniform_point(0, sample_size, np.random.default_rng()),
    )

    compute_se_method = (
        compute_normalized_spectral_entropy if normalized else compute_spectral_entropy
    )

    voronoi_se = np.zeros_like(beta_range)
    total_node_count = 0
    total_edge_count = 0
    samples = generator.generate_ensemble_boundaries(batch_size)
    for sample in samples:
        graph = extract_graph_from_binary_voronoi(sample, junclets_executable_path)
        total_node_count += graph.number_of_nodes()
        total_edge_count += graph.number_of_edges()
        voronoi_se += compute_se_method(graph, beta_range)

    return (
        voronoi_se / batch_size,
        total_node_count / batch_size,
        total_edge_count / batch_size,
    )
