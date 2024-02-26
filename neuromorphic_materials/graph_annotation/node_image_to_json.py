#! /usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np

# tap refers to typed-argument-parser and is listed in pyproject.toml
# noinspection PyPackageRequirements
from tap import Tap


class NodeImageToJsonParser(Tap):
    image_path: Path  # Input image path
    json_path: Path | None = (  # Output path. Omit to use image_path with suffix replaced by .json
        None
    )

    def configure(self) -> None:
        self.add_argument("image_path")
        self.add_argument("-o", "--json_path")


def main() -> None:
    args = NodeImageToJsonParser().parse_args()
    if args.json_path is None:
        args.json_path = args.image_path.with_suffix(".json")

    # Open node image file and extract nodes, swapping coordinates,
    # so they are (x, y) instead of default cv2 / numpy (y, x)
    nodes = {
        tuple(node.tolist())
        for node in np.argwhere(
            cv2.imread(str(args.image_path.resolve()), cv2.IMREAD_GRAYSCALE)
        )[:, [1, 0]]
    }

    with args.json_path.open("w") as save_file:
        json.dump({"nodes": sorted(nodes)}, save_file)


if __name__ == "__main__":
    main()
