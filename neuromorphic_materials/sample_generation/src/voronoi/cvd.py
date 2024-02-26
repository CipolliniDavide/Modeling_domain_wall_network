import typing

import numpy as np

from .site import VoronoiSite


class CentroidalVoronoiDiagram(typing.NamedTuple):
    diagram: np.ndarray
    boundaries: np.ndarray
    sites: list[VoronoiSite]
