#! /usr/bin/env python3

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.pdf import uniform_point
from src.voronoi.arg_parser import VoronoiParser
#from src.voronoi.generator_gpu import VoronoiGeneratorGPU
from src.voronoi.generator import VoronoiGenerator
from varname import nameof


class GenerateVoronoiSamplesParser(VoronoiParser):
    n_samples: int = 1

    def configure(self) -> None:
        # Set default path to current time
        self.add_argument(
            f"--{nameof(self.save_binary)}",
            nargs="?",
            const=Path(f"./data/voronoi/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        )


def main() -> None:
    args = GenerateVoronoiSamplesParser().parse_args()

    tick = time.perf_counter()
    #generator = VoronoiGeneratorGPU(
    generator = VoronoiGenerator(
        args.site_linear_density,
        args.p_horizontal,
        args.horizontal_beta,
        args.sample_size,
        args.n_iterations,
        args.site_update_method,
        uniform_point(0, args.sample_size, np.random.default_rng()),
    )
    print(f"Instantiated VoronoiGenerator in {time.perf_counter() - tick:0.2f} seconds")

    tick = time.perf_counter()

    voronoi_samples = generator.generate_cvds(args.n_samples)
    print(
        f"Computed {args.n_samples} CVT(s) in {time.perf_counter() - tick:0.2f} seconds"
    )

    if args.save_binary is not None:
        args.save_binary.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(voronoi_samples):
            save_path = args.save_binary / f"{idx}.png"
            print(f"Saving binary image to {save_path}")
            cv2.imwrite(str(save_path.resolve()), sample.boundaries)


if __name__ == "__main__":
    main()
