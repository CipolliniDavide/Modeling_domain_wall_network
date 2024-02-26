import json
import logging
from pathlib import Path

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg
from PIL import Image

from ..base_annotator.graph_manager import BaseGraphManager, Point


class GraphManager(BaseGraphManager):
    _sample_image_fig_id: int | None

    _nodes: dict[Point, int]
    _selected_node: Point | None

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

        self._nodes = {}
        self._selected_node = None

    def handle_event(self, event: str, values: dict):
        # Only handle events if sample has nodes
        if self._nodes is None:
            return

        # Handle escape keyboard press
        if event == "esc":
            self._deselect_node()
            return

        # Handle delete keyboard press
        if event == "del":
            self._delete_selected_node()
            return

        # TODO: Handle arrow keys to move selected node

        self._handle_click(values[self.GRAPH_KEY])

    def _draw_node(self, node: Point, color: str) -> int:
        return self._graph.draw_point(node, self.DRAW_POINT_SIZE, color)

    def _handle_click(self, coordinates: Point):
        logging.debug(f"click at {coordinates}")

        clicked_node = self._compute_clicked_node(coordinates)
        if clicked_node is not None:
            logging.debug(f"clicked node {clicked_node}")

            if self._selected_node is not None:
                self._deselect_node()
                return

            self._select_node(clicked_node)
            return

        if self._selected_node is not None:
            self._deselect_node()

        logging.debug(f"creating node {coordinates}")
        self._nodes[coordinates] = self._draw_node(coordinates, self.DRAW_COLOR)

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

        targets = sorted(
            set(self._nodes.keys()).intersection(self._click_area(coordinates))
        )

        # Return the first target after sorting
        return targets[0] if len(targets) > 0 else None

    def _select_node(self, node: Point):
        if node not in self._nodes:
            return

        logging.debug(f"selecting node {node}")
        self._selected_node = node
        # Delete old node figure and redraw in green
        self._graph.delete_figure(self._nodes[self._selected_node])
        self._nodes[self._selected_node] = self._draw_node(
            self._selected_node, self.DRAW_COLOR_SELECTED
        )

    def _deselect_node(self) -> None:
        if self._selected_node is None:
            return

        logging.debug(f"deselecting node {self._selected_node}")
        self._graph.delete_figure(self._nodes[self._selected_node])
        self._nodes[self._selected_node] = self._draw_node(
            self._selected_node, self.DRAW_COLOR
        )
        self._selected_node = None

    def _delete_selected_node(self) -> None:
        if self._selected_node is None:
            return

        self._delete_node(self._selected_node)
        self._selected_node = None

    def _delete_node(self, node: Point):
        if node not in self._nodes:
            return

        logging.debug(f"deleting node {node}")
        self._graph.delete_figure(self._nodes[node])
        del self._nodes[node]

    def open_sample(self, sample_path: Path):
        # Clear all nodes
        for node in self._nodes:
            self._delete_node(node)

        # Clear graph, draw sample file as background, then draw nodes on top
        self._graph.erase()
        self._sample_image_fig_id = self._graph.draw_image(
            data=self._prepare_image(Image.open(sample_path), self.GRAPH_SIZE),
            location=(-0.5, -0.5),
        )

    def save_nodes(self, save_path: Path):
        with save_path.open("w") as save_file:
            json.dump({"nodes": sorted(self._nodes.keys())}, save_file)

    @property
    def graph(self) -> sg.Graph:
        return self._graph
