import json
import logging
from pathlib import Path

import numpy as np

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg
from PIL import Image

from ..base_annotator.graph_manager import BaseGraphManager, Edge, Point


class GraphManager(BaseGraphManager):
    _sample_image_fig_id: int | None
    _nodes_image_fig_id: int | None

    _nodes: set[Point]
    _selected_node: Point | None
    _selected_node_fig_id: int | None

    _edges: dict[Edge, int]
    _selected_edge: Edge | None

    def __init__(self) -> None:
        self._graph = sg.Graph(
            self.GRAPH_SIZE,
            (-0.5, self.SAMPLE_SIZE - 0.5),
            (self.SAMPLE_SIZE - 0.5, -0.5),
            key=self.GRAPH_KEY,
            enable_events=True,
            background_color="black",
        )
        self._sample_image_fig_id = None
        self._nodes_image_fig_id = None

        self._nodes = set()
        self._selected_node = None
        self._selected_node_fig_id = None

        self._edges = {}
        self._selected_edge = None

    def handle_event(self, event: str, values: dict):
        # Only handle events if sample has nodes
        if self._nodes is None:
            return

        # Handle escape keyboard press
        if event == "esc":
            self._deselect_edge()
            self._deselect_node()
            return

        # Handle delete keyboard press
        if event == "del":
            self._delete_selected_edge()
            return

        self._handle_click(values[self.GRAPH_KEY])

    def _draw_edge(self, edge: Edge, color: str) -> int:
        return self._graph.draw_line(edge[0], edge[1], color, self.DRAW_EDGE_WIDTH)

    def _handle_click(self, coordinates: Point):
        logging.debug(f"click at {coordinates}")

        clicked_node = self._compute_clicked_node(coordinates)
        if clicked_node is None:
            self._deselect_node()

            clicked_edge = self._compute_clicked_edge(coordinates)
            prev_selected_edge = self._selected_edge
            self._deselect_edge()
            if clicked_edge is None:
                return

            logging.debug(f"clicked edge {clicked_edge}")

            if prev_selected_edge == clicked_edge:
                return

            self._select_edge(clicked_edge)
            return

        self._deselect_edge()

        logging.debug(f"clicked node {clicked_node}")

        # No node is selected yet, so select the clicked node
        if self._selected_node is None:
            self._select_node(clicked_node)
            return

        # Create an edge between selected node and clicked node, if none exists
        if self._selected_node != clicked_node and not (
            (self._selected_node, clicked_node) in self._edges
            or (clicked_node, self._selected_node) in self._edges
        ):
            logging.debug(f"creating edge {self._selected_node} - {clicked_node}")
            edge = (self._selected_node, clicked_node)
            self._edges[edge] = self._draw_edge(edge, self.DRAW_COLOR)
        self._deselect_node()

    @staticmethod
    def _click_area(coordinates: Point):
        # Locations that are within a 3x3 area around clicked point
        return (
            (x, y)
            for y in range(coordinates[1] - 1, coordinates[1] + 2)
            for x in range(coordinates[0] - 1, coordinates[0] + 2)
        )

    def _compute_clicked_node(self, coordinates: Point) -> Point | None:
        if coordinates in self._nodes:
            return coordinates

        targets = sorted(self._nodes.intersection(self._click_area(coordinates)))

        # Return the first target after sorting
        return targets[0] if len(targets) > 0 else None

    def _find_edge_at_coordinates(self, coordinates: Point):
        # Get all figures at coordinates
        figs = list(self._graph.get_figures_at_location(coordinates))

        # Remove background images and selected node
        try:
            figs.remove(self._sample_image_fig_id)
            figs.remove(self._nodes_image_fig_id)
            figs.remove(self._selected_node_fig_id)
        except ValueError:
            pass  # No problem if the ids are not in the list

        return (
            None
            if len(figs) == 0
            else list(self._edges.keys())[list(self._edges.values()).index(figs[0])]
        )

    def _compute_clicked_edge(self, coordinates: Point) -> Edge | None:
        maybe_edge = self._find_edge_at_coordinates(coordinates)
        if maybe_edge is not None:
            return maybe_edge

        # Check neighbouring pixels if the pixel clicked does not contain an edge
        for location in self._click_area(coordinates):
            maybe_edge = self._find_edge_at_coordinates(location)
            if maybe_edge is not None:
                return maybe_edge

        return None

    def _select_edge(self, edge: Edge):
        if edge not in self._edges:
            return

        logging.debug(f"selecting edge {edge}")
        self._selected_edge = edge
        # Delete old edge figure and redraw in green
        self._graph.delete_figure(self._edges[self._selected_edge])
        self._edges[self._selected_edge] = self._draw_edge(
            self._selected_edge, self.DRAW_COLOR_SELECTED
        )

    def _deselect_edge(self) -> None:
        if self._selected_edge is None:
            return

        logging.debug(f"deselecting edge {self._selected_edge}")
        self._graph.delete_figure(self._edges[self._selected_edge])
        self._edges[self._selected_edge] = self._draw_edge(
            self._selected_edge, self.DRAW_COLOR
        )
        self._selected_edge = None

    def _delete_selected_edge(self) -> None:
        if self._selected_edge is None:
            return

        self._delete_edge(self._selected_edge)
        self._selected_edge = None

    def _delete_edge(self, edge: Edge):
        if edge not in self._edges:
            return

        logging.debug(f"deleting edge {edge}")
        self._graph.delete_figure(self._edges[edge])
        del self._edges[edge]

    def _select_node(self, node: Point):
        logging.debug(f"selecting node {node}")
        self._selected_node = node
        self._selected_node_fig_id = self._graph.draw_point(
            node, self.DRAW_POINT_SIZE, self.DRAW_COLOR_SELECTED
        )

    def _deselect_node(self) -> None:
        if self._selected_node is None:
            return

        logging.debug(f"deselecting node {self._selected_node}")
        self._selected_node = None
        self._graph.delete_figure(self._selected_node_fig_id)

    def open_sample(self, sample_path: Path, nodes_path: Path):
        # Load nodes from nodes file
        with nodes_path.open("r") as nodes_file:
            self._nodes = {tuple(node) for node in json.load(nodes_file)["nodes"]}

        # Delete all edges
        for edge in self._edges:
            self._delete_edge(edge)

        # Clear graph, draw sample file as background, then draw nodes on top
        self._graph.erase()
        self._sample_image_fig_id = self._graph.draw_image(
            data=self._prepare_image(Image.open(sample_path), self.GRAPH_SIZE),
            location=(-0.5, -0.5),
        )

        # Create node image from node locations by setting red and alpha channel to 255 for each node location
        nodes_image = np.zeros((self.SAMPLE_SIZE, self.SAMPLE_SIZE, 4), dtype=np.uint8)
        for node in self._nodes:
            nodes_image[*node[::-1], (0, 3)] = 255
        self._nodes_image_fig_id = self._graph.draw_image(
            data=self._prepare_image(Image.fromarray(nodes_image), self.GRAPH_SIZE),
            location=(-0.5, -0.5),
        )

    def save_nodes_and_edges(self, save_path: Path):
        with save_path.open("w") as save_file:
            json.dump(
                {"nodes": sorted(self._nodes), "edges": sorted(self._edges.keys())},
                save_file,
            )

    @property
    def graph(self) -> sg.Graph:
        return self._graph
