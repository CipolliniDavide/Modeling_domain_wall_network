#! /usr/bin/env python3

import time

import numpy as np
from matplotlib import pyplot as plt
from src.pdf import uniform_point
from src.voronoi.arg_parser import VoronoiParser
from src.voronoi.generator import VoronoiGenerator

from neuromorphic_materials.sample_generation.src.voronoi.generator_gpu import (
    VoronoiGeneratorGPU,
)


def main() -> None:
    parser = VoronoiParser()
    args = parser.parse_args()

    n_cvds = 5
    point_gen = uniform_point(0, args.sample_size, np.random.default_rng())

    tick = time.perf_counter()
    generator = VoronoiGenerator(
        args.site_linear_density,
        args.p_horizontal,
        args.horizontal_beta,
        args.sample_size,
        args.n_iterations,
        args.site_update_method,
        point_gen,
    )
    generator.generate_ensemble_boundaries(n_cvds)
    tock = time.perf_counter()
    print(
        f"Computed {n_cvds} NumPy CVDs in {tock - tick:0.2f} seconds,"
        f" {(tock - tick) / n_cvds:0.2f} seconds per CVD"
    )

    tick = time.perf_counter()
    generator = VoronoiGeneratorGPU(
        args.site_linear_density,
        args.p_horizontal,
        args.horizontal_beta,
        args.sample_size,
        args.n_iterations,
        args.site_update_method,
        point_gen,
    )
    generator.generate_ensemble_boundaries(n_cvds)
    tock = time.perf_counter()
    print(
        f"Computed {n_cvds} CuPy CVDs in {tock - tick:0.2f} seconds,"
        f" {(tock - tick) / n_cvds:0.2f} seconds per CVD"
    )


if __name__ == "__main__":
    main()
