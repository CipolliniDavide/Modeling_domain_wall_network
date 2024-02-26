#! /usr/bin/env python3
import logging
from abc import ABC, abstractmethod

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg

from .graph_manager import TGraphManager


class BaseAnnotator(ABC):
    _window: sg.Window
    _graph_manager: TGraphManager

    def start(self) -> None:
        # Cannot start if window was already closed once
        if self._window.is_closed():
            return

        while True:
            event, values = self._window.read()
            # See if user wants to quit or window was closed
            if event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, "quit"):
                if sg.popup_yes_no("Really quit?", keep_on_top=True) == "Yes":
                    break
            elif event == "open":
                self._open_sample()
            elif event == "save":
                self._save()
            elif event in (self._graph_manager.GRAPH_KEY, "esc", "del"):
                self._graph_manager.handle_event(event, values)
            else:
                logging.debug(f"event not handled: {event}\nvalues: {values}")

        self._window.close()

    @abstractmethod
    def _open_sample(self) -> None:
        pass

    @abstractmethod
    def _save(self) -> None:
        pass
