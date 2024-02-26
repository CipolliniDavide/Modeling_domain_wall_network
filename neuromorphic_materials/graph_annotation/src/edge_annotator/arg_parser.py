from pathlib import Path

# tap refers to typed-argument-parser and is listed in pyproject.toml
# noinspection PyPackageRequirements
from tap import Tap


class EdgeAnnotatorParser(Tap):
    sample_files: tuple[Path, Path] | None = None  # Sample image and nodes json
    debug: bool = False  # Enable debug logging
