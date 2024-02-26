from pathlib import Path

import cv2
import numpy as np

from ..point import SizeTuple
from .line import SampleLine


class SampleImage:
    def __init__(self, image_size: SizeTuple) -> None:
        self._image: np.ndarray = np.zeros(image_size, dtype=np.uint8)
        self._color = 255

    def draw_line(self, line: SampleLine) -> None:
        cv2.line(
            self._image,
            line.start_point.to_int_tuple(),
            line.end_point.to_int_tuple(),
            self._color,
            line.thickness,
            cv2.LINE_AA,
        )

    def save(self, save_path: Path) -> bool:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(save_path.resolve()), self._image)
