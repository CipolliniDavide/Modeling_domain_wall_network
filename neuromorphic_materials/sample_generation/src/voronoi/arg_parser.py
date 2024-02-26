from pathlib import Path
from typing import Literal, TypeAlias

from tap import Tap
from varname import nameof

LocationUpdateMethod: TypeAlias = Literal["centroid", "hard_core"]


class VoronoiParser(Tap):
    sample_size: int = 128
    site_linear_density: float = 0.9229
    p_horizontal: float = 0.4285
    horizontal_beta: float = 2.6
    n_iterations: int = 10
    rng_seed: int = 0xDEADCAFE
    site_update_method: LocationUpdateMethod = "centroid"
    save_binary: Path | None = None  # Save binary sample image to given path

    def configure(self) -> None:
        # Set default save location if save binary is given without argument
        self.add_argument(
            f"--{nameof(self.save_binary)}", nargs="?", const=Path("./data/voronoi")
        )
