import typing
from math import atan2, degrees, sqrt
from typing import Self, TypeAlias

import numpy as np
import numpy.typing as npt

from .rounding import int_round_half_away_from_zero

PointTuple: TypeAlias = tuple[float, float]
ImagePoint: TypeAlias = tuple[int, int]
SizeTuple: TypeAlias = ImagePoint


class Point(typing.NamedTuple):
    x: float
    y: float

    @classmethod
    def from_tuple(cls, tup: PointTuple) -> Self:
        return cls(*tup)

    @classmethod
    def from_np_arr(cls, arr: npt.NDArray[np.float_]) -> Self:
        """
        Convert the numpy array (must have 2 elements) to a list with python
        scalars, and destructure it into an x and y value for the Point
        """
        return cls(*arr.tolist())

    def to_tuple(self) -> PointTuple:
        return self.x, self.y

    def to_int_tuple(self) -> ImagePoint:
        return int_round_half_away_from_zero(self.x), int_round_half_away_from_zero(
            self.y
        )

    def dist_to_point(self, other: Self) -> float:
        return sqrt(abs(other.x - self.x) ** 2 + abs(other.y - self.y) ** 2)

    def angle_to_point(self, other: Self) -> float:
        return degrees(atan2(other.y - self.y, other.x - self.x))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


Edge: TypeAlias = tuple[Point, Point]
