#! /usr/bin/env python3
from pathlib import Path

import cv2
import networkx as nx
from matplotlib import pyplot as plt
from tap import Tap
from varname import nameof

import os
import sys
sys.path.insert(1, '{:s}/'.format(os.getcwd().strip('neuromorphic_materials')))

from neuromorphic_materials.junction_graph_extraction.extract_graph_voronoi import (
    extract_graph_from_binary_voronoi,
)


class ExtractGraphVoronoiParser(Tap):
    input_file: Path = None  # Input Voronoi sample image
    graph_file: Path = None  # Output graph file path
    junclets_executable_path: Path = Path(  # Path to junclets executable
        "./beyondOCR_junclets/my_junc"
    )

    def configure(self) -> None:
        self.add_argument(nameof(self.input_file))
        self.add_argument(nameof(self.graph_file))
        self.add_argument(nameof(self.junclets_executable_path))


def main() -> None:
    parser = ExtractGraphVoronoiParser()
    args = parser.parse_args()

    image = cv2.imread(str(args.input_file.resolve()), cv2.IMREAD_GRAYSCALE)

    graph = extract_graph_from_binary_voronoi(image, args.junclets_executable_path)

    if graph is None:
        print("Junclets encountered an error")
        return

    args.graph_file.parent.mkdir(exist_ok=True, parents=True)
    nx.write_graphml(graph, args.graph_file)


if __name__ == "__main__":
    main()
