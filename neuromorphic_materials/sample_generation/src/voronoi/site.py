import typing
from typing import Self, TypeAlias

import numpy as np
import numpy.typing as npt

from ..point import Point, PointTuple

VoronoiSiteTuple: TypeAlias = tuple[float, float, float]


class VoronoiSite(typing.NamedTuple):
    x: float
    y: float
    beta: float

    @classmethod
    def from_tuple(cls, tup: VoronoiSiteTuple) -> Self:
        return cls(*tup)

    @classmethod
    def from_point(cls, point: Point, beta: float) -> Self:
        return cls(point.x, point.y, beta)

    @classmethod
    def from_np_arr(cls, arr: npt.NDArray[np.float_]) -> Self:
        """
        Convert the numpy array to a list with python scalars,
        and destructure it into an x and y value for the Point
        """
        assert (
            arr.shape == (3,) and arr.dtype == np.float_
        ), "Array must have 3 elements and datatype float64"
        return cls(*arr.tolist())

    def to_tuple(self) -> VoronoiSiteTuple:
        return self.x, self.y, self.beta

    def coordinate_tuple(self) -> PointTuple:
        return self.x, self.y

    def dist_to_point(self, other: Self) -> float:
        return max(self.beta * abs(self.x - other.x), abs(self.y - other.y) / self.beta)

    def __str__(self) -> str:
        return f"({self.x}, {self.y}), beta={self.x}"
