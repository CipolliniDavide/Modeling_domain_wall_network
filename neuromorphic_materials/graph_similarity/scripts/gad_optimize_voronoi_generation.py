#! /usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt

# sys.path.insert(1, '{:s}/PycharmProjects/Bfo_network/'.format(os.path.expanduser('~')))
sys.path.insert(1, '{:s}/'.format(os.getcwd()))

import time
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pygad
from tap import Tap
from varname import nameof

from neuromorphic_materials.graph_similarity.spectral_entropy import (
    compute_normalized_spectral_entropy,
    compute_spectral_entropy,
)
from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)
from neuromorphic_materials.sample_generation.src.pdf import uniform_point
from neuromorphic_materials.sample_generation.src.voronoi.generator import (
    VoronoiGenerationError,
    VoronoiGenerator,
)
from helpers_graph import (connected_components, merge_nodes_by_distance, remove_self_loops)
from helpers_plot import fitness_plot

class GadOptimizeVoronoiGenerationParser(Tap):
    bfo_graph_dir: Path = None  # Directory with BFO GraphML files
    junclets_executable_path: Path = None  # Path to junclets executable
    save_dir: Path = Path(f"./data/voronoi_gads")  # Directory where to save optimization results
    num_generations: int = 51
    processes: int = 8
    n_iterations: int = 20 # number of iterations of centroid Voronoi tessellation
    merge_nodes_epsilon: float = 1.1  # Hyperparameter used in the automatic graph extraction to merge nodes that are in a radius smaller r<epsilon
    beta_range: tuple[
        int, int, int
    ] = (  # Beta range in log10 space, last number is number of steps
        -3,
        4,
        300,
    )
    use_se_gradient: bool = False
    use_normalized_se: bool = False

    def configure(self) -> None:
        self.add_argument(nameof(self.bfo_graph_dir))
        self.add_argument(nameof(self.junclets_executable_path))
        self.add_argument(nameof(self.save_dir))


class VoronoiGenerationGA(pygad.GA):
    save_dir: Path
    junclets_executable_path: Path
    bfo_ensemble_eigenvalues: np.ndarray
    bfo_ensemble_mean_se: np.ndarray
    beta_range: np.ndarray
    ensemble_size: int
    sample_size: int
    n_iterations: int
    use_se_gradient: bool
    use_normalized_se: bool
    timestamp: float
    merge_nodes_epsilon: float
    saved_solutions: list[np.ndarray] = []
    saved_fitnesses: list[np.ndarray] = []


def fitness_function_log_likelihood(
    ga: VoronoiGenerationGA, solution: np.ndarray, _: int
) -> float:
    site_linear_density: float
    p_horizontal: float
    horizontal_beta: float
    (site_linear_density, p_horizontal, horizontal_beta) = solution

    generator = VoronoiGenerator(
        site_linear_density,
        p_horizontal,
        horizontal_beta,
        ga.sample_size,
        ga.n_iterations,
        "centroid",
        uniform_point(0, ga.sample_size, np.random.default_rng()),
    )

    # tick = time.perf_counter()

    try:
        # TODO: Reduce overhead in GA (parallel) sample generation
        voronoi_boundaries = generator.generate_ensemble_boundaries(ga.ensemble_size)
    except VoronoiGenerationError:
        print(f"Voronoi Generation Error occurred")
        return 0.0
    voronoi_se = np.zeros_like(ga.beta_range)
    for boundaries in voronoi_boundaries:
        graph = extract_graph_from_binary_voronoi(
            boundaries, ga.junclets_executable_path
        )
        # Fitness is 0 if graph has no nodes or edges
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            print(f"Graph has 0 nodes")
            return 0.0
        voronoi_se += compute_normalized_spectral_entropy(graph, ga.beta_range)

    # tock = time.perf_counter()
    # print(
    #     f"Computed {ga.ensemble_size} SEs in {tock - tick:0.2f}s,"
    #     f" {(tock - tick) / ga.ensemble_size:0.2f}s per SE"
    # )

    # TODO: Decide on good loss function
    return 1 / ((ga.bfo_ensemble_mean_se - voronoi_se / ga.ensemble_size) ** 2).sum()


def fitness_function_se_sse(
    ga: VoronoiGenerationGA, solution: np.ndarray, _: int
) -> float:
    site_linear_density: float
    p_horizontal: float
    horizontal_beta: float
    (site_linear_density, p_horizontal, horizontal_beta) = solution

    generator = VoronoiGenerator(
        site_linear_density,
        p_horizontal,
        horizontal_beta,
        ga.sample_size,
        ga.n_iterations,
        "centroid",
        uniform_point(0, ga.sample_size, np.random.default_rng()),
    )

    # tick = time.perf_counter()

    compute_se_method = (
        compute_normalized_spectral_entropy
        if ga.use_normalized_se
        else compute_spectral_entropy
    )

    try:
        # TODO: Reduce overhead in GA (parallel) sample generation
        voronoi_boundaries = generator.generate_ensemble_boundaries(ga.ensemble_size)
    except VoronoiGenerationError:
        print(f"Voronoi Generation Error occurred")
        return 0.0
    voronoi_se = np.zeros_like(ga.beta_range)
    for boundaries in voronoi_boundaries:
        try:
            # Extract graph from Voronoi
            graph = extract_graph_from_binary_voronoi(boundaries, ga.junclets_executable_path)
            # print(graph.number_of_nodes(), graph.number_of_edges())
            # Merge nodes closer than epsilon
            merged_graph = merge_nodes_by_distance(graph, epsilon=ga.merge_nodes_epsilon, return_removed_nodes=False)
            # merged_graph, num_rem_nodes = merge_nodes_by_distance(graph, epsilon=ga.merge_nodes_epsilon, return_removed_nodes=True)
            # print(f'Removed nodes {len(num_rem_nodes)}')

            # Select connected component
            graph = connected_components(merged_graph)[0]
            # print('Before self-loop: ', graph.number_of_nodes(), graph.number_of_edges())

            # Remove any self-loop:
            graph = remove_self_loops(graph)
            # print('After self-loop: ', graph.number_of_nodes(), graph.number_of_edges())

            # Fitness is 0 if graph has no nodes or edges
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                print(f"Graph has 0 nodes")
                return 0.0
            voronoi_se += compute_se_method(graph, ga.beta_range)
        except:
            print('Error with juncs extraction.')
    # tock = time.perf_counter()
    # print(
    #     f"Computed {ga.ensemble_size} SEs in {tock - tick:0.2f}s,"
    #     f" {(tock - tick) / ga.ensemble_size:0.2f}s per SE"
    # )

    # TODO: Decide on good loss function
    voronoi_se = np.gradient(voronoi_se) if ga.use_se_gradient else voronoi_se
    return 1 / ((ga.bfo_ensemble_mean_se - voronoi_se / ga.ensemble_size) ** 2).sum()


def on_fitness(ga: VoronoiGenerationGA, fitness: np.ndarray):
    # Save the top-10 fittest solutions and their fitness values
    top_ten_idx = np.argpartition(fitness, -10)[-10:]
    ga.saved_solutions.append(ga.population.copy()[top_ten_idx])
    ga.saved_fitnesses.append(fitness.copy()[top_ten_idx])


def on_generation(ga: VoronoiGenerationGA):
    solution, fitness, _ = ga.best_solution(ga.last_generation_fitness)
    site_linear_density, p_horizontal, horizontal_beta, *_ = solution
    new_timestamp = time.perf_counter()
    print(
        f"Generation {ga.generations_completed} took"
        f" {(new_timestamp - ga.timestamp)/60:0.2f}min\nBest solution fitness:"
        f" {fitness:0.4f}\nBest solution params:\n  {site_linear_density=:0.4f}\n "
        f" {p_horizontal=:0.4f}\n  {horizontal_beta=:0.4f}\n"
    )

    # print('best solution')
    # print(ga.best_solutions[-1])
    # print('\n\n')

    ga.timestamp = new_timestamp

    # save_dir = Path(f"./data/voronoi_gads1")
    ga.save_dir.mkdir(parents=True, exist_ok=True)
    fitness_plot(ga, save_path=str(ga.save_dir / "plotfitness"),
                 figname='_maxfitness', figformat='.png',
                 figsize=(6, 4),
                 show=False, max_fitness=True)
    fitness_plot(ga, save_path=str(ga.save_dir / "plotfitness"),
                 figname='_meanfitness', figformat='.png',
                 figsize=(6, 4),
                 show=False, max_fitness=False)


def main() -> None:
    args = GadOptimizeVoronoiGenerationParser().parse_args()

    print('\nArgs:\n\t', args, '\n')

    # Select connected component
    bfo_graphs = [connected_components(nx.read_graphml(graph_file))[0]
                  for graph_file in args.bfo_graph_dir.glob("*.graphml")]

    # Merge nodes closer than epsilon & remove self-loops if any
    bfo_graphs = [remove_self_loops(merge_nodes_by_distance(g, epsilon=args.merge_nodes_epsilon, return_removed_nodes=False))
                  for g in bfo_graphs]

    compute_se_method = (
        compute_normalized_spectral_entropy
        if args.use_normalized_se
        else compute_spectral_entropy
    )
    beta_range = np.logspace(*args.beta_range)
    bfo_se = np.zeros_like(beta_range)
    for graph in bfo_graphs:
        bfo_se += compute_se_method(graph, beta_range)
    bfo_ensemble_mean_se = bfo_se / len(bfo_graphs)

    ga = VoronoiGenerationGA(
        num_generations=args.num_generations,
        num_parents_mating=24,
        fitness_func=fitness_function_se_sse,
        sol_per_pop=24 * 4,
        num_genes=3,
        gene_space=[
            {"low": 0.1, "high": 1.0},  # site_linear_density
            {"low": 0.0, "high": 1.0},  # p_horizontal
            {"low": 1.0, "high": 3.0},  # horizontal_beta
        ],
        # gene_type=[np.float32, np.float32, np.float32],
        parent_selection_type="rank",  # use "sus" or "rank"
        parallel_processing=("process", args.processes),
        on_fitness=on_fitness,
        on_generation=on_generation,
        # save_solutions=True,
        save_best_solutions=True,
        stop_criteria=["saturate_100"],
        # mutation_type="adaptive",  # Not parallelised
        # mutation_num_genes=(2, 1),
        mutation_probability=0.1,
        keep_parents=0,
        keep_elitism=0,
    )

    # ga.bfo_ensemble_eigenvalues =
    ga.bfo_ensemble_mean_se = (
        np.gradient(bfo_ensemble_mean_se)
        if args.use_se_gradient
        else bfo_ensemble_mean_se
    )
    ga.save_dir = args.save_dir
    ga.junclets_executable_path = args.junclets_executable_path
    ga.beta_range = beta_range
    ga.use_se_gradient = args.use_se_gradient
    ga.use_normalized_se = args.use_normalized_se
    ga.ensemble_size = len(bfo_graphs) # 25
    ga.sample_size = 128
    ga.n_iterations = args.n_iterations
    ga.timestamp = time.perf_counter()
    ga.merge_nodes_epsilon = args.merge_nodes_epsilon

    print('Len dataset', len(bfo_graphs))
    print(np.shape(bfo_se), np.max(bfo_ensemble_mean_se), np.min(bfo_ensemble_mean_se))
    # print('Use normalized sse', args.use_normalized_se)
    # print('path junc ', args.junclets_executable_path)
    # print('Use se grad', args.use_se_gradient)
    # print(f'Merge node espilon: {args.merge_nodes_epsilon}, {ga.merge_nodes_epsilon}')

    tick = time.perf_counter()
    ga.run()
    tock = time.perf_counter()
    # print(f"GA took {tock - tick:0.2f} seconds")
    print(f"GA took {(tock - tick)/60/24:0.2f} hours")

    # save_dir = Path(f"../../../data/voronoi_gads1")
    # save_dir = Path(f"./data/voronoi_gads1")
    save_dir = ga.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    ga.save(str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S")))

    print(ga.best_solutions[-1])

    # fig = ga.plot_fitness()
    # fig.savefig(str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S_plotfitness.svg")))
    # plt.close()

    # fig = ga.plot_new_solution_rate()
    # fig.savefig(str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S_newsolutionrate.svg")))
    # plt.close(fig=fig)

    # fig = ga.plot_genes(graph_type="boxplot", solutions='all')
    # fig.savefig(str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S_genes.svg")))
    # plt.close(fig=fig)

    fitness_plot(ga, save_path=str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S_plotfitness")),
                 figname='_maxfitness', figformat='.pdf',
                 figsize=(6, 4),
                 show=False, max_fitness=True)
    fitness_plot(ga, save_path=str(save_dir / datetime.now().strftime("%Y%m%d-%H%M%S_plotfitness")),
                 figname='_meanfitness', figformat='.pdf',
                 figsize=(6, 4),
                 show=False, max_fitness=False)
    plt.close()


if __name__ == "__main__":
    main()
