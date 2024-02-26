#! /usr/bin/env python3
from pathlib import Path

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg

from ..base_annotator.base_annotator import BaseAnnotator
from .graph_manager import GraphManager


class EdgeAnnotator(BaseAnnotator):
    def __init__(self) -> None:
        sg.theme("Default 1")
        self._graph_manager = GraphManager()
        self._window = sg.Window(
            "Edge Annotator",
            [
                [self._graph_manager.graph],
                [
                    sg.Button("Open sample", key="open"),
                    sg.Button("Save edges", key="save"),
                    sg.Button("Quit", key="quit"),
                ],
            ],
            enable_close_attempted_event=True,
            finalize=True,
        )
        self._window.bind("<Escape>", "esc")
        self._window.bind("<Delete>", "del")

    def _open_sample(self) -> None:
        sample_filename = sg.popup_get_file(
            "Select sample image file", "Select sample", file_types=(("PNG", ".png"),)
        )

        if not sample_filename:
            sg.popup_ok(
                "No file selected, cannot open sample", title="Cannot open sample"
            )
            return

        nodes_filename = sg.popup_get_file(
            "Select node JSON file", "Select nodes", file_types=(("JSON", ".json"),)
        )

        if not sample_filename:
            sg.popup_ok(
                "No file selected, cannot open nodes", title="Cannot open nodes"
            )
            return

        self._graph_manager.open_sample(Path(sample_filename), Path(nodes_filename))

    def open_sample_from_paths(self, sample_path: Path, nodes_path: Path):
        self._graph_manager.open_sample(sample_path, nodes_path)

    def _save(self) -> None:
        filename = sg.popup_get_file(
            "Select edges save file",
            "Save edges",
            default_extension=".json",
            save_as=True,
            no_window=True,
        )

        if not filename:
            sg.popup_ok(
                "No file selected, cannot save edges", title="Cannot save edges"
            )
            return

        save_path = Path(filename)
        self._graph_manager.save_nodes_and_edges(save_path)
        sg.popup_ok(
            f"Saved nodes and edges to\n{save_path.name}",
            title="Saved nodes and edges",
            keep_on_top=True,
        )
