from pathlib import Path
from typing import TypeAlias, TypeVar

from tap import Tap


class BaseSampleParser(Tap):
    save_dir: Path  # Where the newly created sample is saved
    num_samples: int = 1  # The number of samples to create
    sample_size: int = 256  # Sample side length in pixels
    line_thickness: int = 1  # Thickness of the lines drawn in the sample

    def configure(self) -> None:
        self.add_argument("save_dir")
        self.add_argument("-n", "--num_samples")


TSampleParser: TypeAlias = TypeVar("TSampleParser", bound=BaseSampleParser)
