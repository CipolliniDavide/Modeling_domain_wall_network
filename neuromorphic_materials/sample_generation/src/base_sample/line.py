from typing import Self, TypedDict

import numpy as np

from ..point import Point, PointTuple, SizeTuple


class SampleLineDict(TypedDict):
    start_point: PointTuple
    angle: float
    length: float
    thickness: int


class SampleLine:
    def __init__(
        self, start_point: Point, angle: float, length: float, thickness: int
    ) -> None:
        self._start_point: Point = start_point
        self._angle: float = angle % 360
        self._length: float = length
        self._thickness: int = thickness

    def fit_to_sample(self, sample_size: SizeTuple) -> bool:
        img_x, img_y = sample_size
        end_x, end_y = self.end_point
        # Only change anything if the line endpoint is outside the given image size
        if 0 < end_x < img_x and 0 < end_y < img_y:
            return False

        start_x, start_y = self._start_point

        # Line equation representation of the image borders in the
        # form ax + by + c = 0. The tuples represent a, b, and c.
        borders = [(1, 0, 0), (1, 0, -img_x + 1), (0, 1, 0), (0, 1, -img_y + 1)]
        for border_a, border_b, border_c in borders:
            # Get the same line equation for the line drawn
            if (start_x - end_x) == 0:
                # Special case for vertical lines, as these have an undefined slope
                line_a = 1
                line_b = 0
                line_c = -start_x
            else:
                m = (start_y - end_y) / (start_x - end_x)
                line_a = -m
                line_b = 1
                line_c = m * start_x - start_y

            # Compute intersection point of the line with each border
            denominator = border_a * line_b - line_a * border_b
            if denominator == 0:
                # no intersection exists if the denominator is zero, skip this border
                continue
            intersect_x = (border_b * line_c - line_b * border_c) / denominator
            intersect_y = (border_c * line_a - line_c * border_a) / denominator
            # Update the length if the intersection point of
            # the two lines falls on the image border
            # TODO: Change start < intersect < end to work for end < start
            if (
                border_a == 1
                and 0 <= intersect_y <= img_y - 1
                and (start_x <= intersect_x <= end_x or end_x <= intersect_x <= start_x)
            ) or (
                border_a == 0
                and 0 <= intersect_x <= img_x - 1
                and (start_y <= intersect_y <= end_y or end_y <= intersect_y <= start_y)
            ):
                old_length = self._length
                new_end_point = Point(intersect_x, intersect_y)
                self._length = self._start_point.dist_to_point(new_end_point)
                # Recompute angle to ensure correct end-point computation
                self._angle = self._start_point.angle_to_point(new_end_point)

                # Assert new length is reduced, or does not modify the endpoint
                assert self._length <= old_length or self.end_point == (
                    intersect_x,
                    intersect_y,
                ), (
                    f"New length ({self._length}) must be smaller than or equal to old"
                    " length"
                )
                return True

        return False

    def intersection_point(self, line: Self) -> Point | None:
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        x3, y3 = line.start_point
        x4, y4 = line.end_point

        x1x2 = x1 - x2
        x1x3 = x1 - x3
        x3x4 = x3 - x4
        y1y2 = y1 - y2
        y1y3 = y1 - y3
        y3y4 = y3 - y4
        denominator = x1x2 * y3y4 - y1y2 * x3x4
        if denominator == 0:
            return None

        t = (x1x3 * y3y4 - y1y3 * x3x4) / denominator
        u = (x1x3 * y1y2 - y1y3 * x1x2) / denominator

        # There is only an intersection if t and u are between 0 and 1
        if t < 0 or t > 1 or u < 0 or u > 1:
            return None

        return Point(x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def to_dict(self) -> SampleLineDict:
        return SampleLineDict(
            start_point=self._start_point.to_tuple(),
            angle=self._angle,
            length=self._length,
            thickness=self._thickness,
        )

    @classmethod
    def from_dict(cls, data: SampleLineDict) -> Self:
        return cls(
            Point.from_tuple(data["start_point"]),
            data["angle"],
            data["length"],
            data["thickness"],
        )

    @classmethod
    def between_points(
        cls, start_point: Point, end_point: Point, thickness: int
    ) -> Self:
        return cls(
            start_point,
            start_point.angle_to_point(end_point),
            start_point.dist_to_point(end_point),
            thickness,
        )

    @property
    def start_point(self) -> Point:
        return self._start_point

    @property
    def end_point(self) -> Point:
        angle_rad = np.radians(self._angle)
        return Point(
            self._start_point.x + self._length * np.cos(angle_rad),
            self._start_point.y + self._length * np.sin(angle_rad),
        )

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self._angle = angle % 360

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, length: float) -> None:
        self._length = length

    @property
    def thickness(self) -> int:
        return self._thickness

    def __str__(self) -> str:
        return (
            f"Start point: {self._start_point}\nEnd point: {self.end_point}\n"
            f"Length: {self._length}\nAngle: {self._angle}"
        )
