#! /usr/bin/env python3
import json
from pathlib import Path

import networkx as nx
from src.base_annotator.graph_manager import Edge, Point
from tap import Tap
from varname import nameof


class AnnotationToGraphParser(Tap):
    annotation_dir: Path = None
    graph_save_dir: Path = None

    def configure(self) -> None:
        self.add_argument(nameof(self.annotation_dir))
        self.add_argument(nameof(self.graph_save_dir))


def generate_graph_from_annotation(annotation: dict) -> nx.Graph:
    # Nodes consist of the coordinate tuple and a dictionary with node coordinates
    #  which adds the coordinates as attributes when saved as GraphML
    nodes: list[tuple[Point, dict]] = [
        (tuple(node), dict(x=node[0], y=node[1])) for node in annotation["nodes"]
    ]
    edges: list[Edge] = [
        (tuple(edge[0]), tuple(edge[1])) for edge in annotation["edges"]
    ]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def annotation_to_graph(annotation_path: Path, save_path: Path):
    with annotation_path.open("r") as annotation_file:
        annotation = json.load(annotation_file)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    nx.write_graphml(generate_graph_from_annotation(annotation), save_path)


def main() -> None:
    args = AnnotationToGraphParser().parse_args()

    if not args.annotation_dir.is_dir() or not args.graph_save_dir.is_dir():
        print("Please provide an annotation file directory and a graph save directory")
        return

    for annotation_file in args.annotation_dir.glob("*.json"):
        annotation_to_graph(
            annotation_file,
            (args.graph_save_dir / annotation_file.stem).with_suffix(".graphml"),
        )


if __name__ == "__main__":
    main()
