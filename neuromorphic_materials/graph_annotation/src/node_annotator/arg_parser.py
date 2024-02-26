from pathlib import Path

# tap refers to typed-argument-parser and is listed in pyproject.toml
# noinspection PyPackageRequirements
from tap import Tap


class NodeAnnotatorParser(Tap):
    sample_file: Path | None = None  # Sample image
    debug: bool = False  # Enable debug logging
