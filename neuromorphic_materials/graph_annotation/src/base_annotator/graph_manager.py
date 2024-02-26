import io
from abc import ABC, abstractmethod
from typing import ClassVar, TypeAlias, TypeVar

# pep8-naming rules sometimes work in unwanted ways
# noinspection PyPep8Naming
import PySimpleGUI as sg

# Resampling exists but the typeshed stubs
# shipped with PyCharm don't include it
# noinspection PyUnresolvedReferences
from PIL.Image import Image, Resampling

Point: TypeAlias = tuple[int, int]
Edge: TypeAlias = tuple[Point, Point]


class BaseGraphManager(ABC):
    # TODO: Add support for variable size graph and crops
    GRAPH_KEY: ClassVar[str] = "graph"
    GRAPH_SIZE: ClassVar[tuple[int, int]] = (512, 512)
    SAMPLE_SIZE: ClassVar[int] = 128
    DRAW_COLOR: ClassVar[str] = "#FF0000"
    DRAW_COLOR_SELECTED: ClassVar[str] = "#00FF00"
    DRAW_POINT_SIZE: ClassVar[int] = 1
    DRAW_EDGE_WIDTH: ClassVar[int] = 2

    _graph: sg.Graph

    @abstractmethod
    def handle_event(self, event: str, values: dict):
        pass

    @staticmethod
    def _prepare_image(image: Image, size: Point) -> bytes:
        bio = io.BytesIO()
        image.resize(size, Resampling.NEAREST).save(bio, format="PNG")
        return bio.getvalue()


TGraphManager: TypeAlias = TypeVar("TGraphManager", bound=BaseGraphManager)
