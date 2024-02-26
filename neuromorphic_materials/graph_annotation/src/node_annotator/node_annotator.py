#! /usr/bin/env python3
from pathlib import Path

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg

from ..base_annotator.base_annotator import BaseAnnotator
from .graph_manager import GraphManager


class NodeAnnotator(BaseAnnotator):
    def __init__(self) -> None:
        sg.theme("Default 1")
        self._graph_manager = GraphManager()
        # TODO: Add support for variable size graph and crops
        self._window = sg.Window(
            "Node Annotator",
            [
                [self._graph_manager.graph],
                [
                    sg.Button("Open sample", key="open"),
                    sg.Button("Save nodes", key="save"),
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

        self._graph_manager.open_sample(Path(sample_filename))

    def open_sample_from_path(self, sample_path: Path):
        self._graph_manager.open_sample(sample_path)

    def _save(self) -> None:
        filename = sg.popup_get_file(
            "Select nodes save file",
            "Save nodes",
            default_extension=".json",
            save_as=True,
            no_window=True,
        )

        if not filename:
            sg.popup_ok(
                "No file selected, cannot save nodes", title="Cannot save nodes"
            )
            return

        save_path = Path(filename)
        self._graph_manager.save_nodes(save_path)
        sg.popup_ok(f"Saved nodes to\n{save_path.name}", keep_on_top=True)
