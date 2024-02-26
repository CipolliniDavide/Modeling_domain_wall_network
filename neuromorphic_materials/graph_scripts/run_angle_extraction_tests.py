#! /usr/bin/env python3
import multiprocessing
from multiprocessing import Pool
from pathlib import Path

from helpers.angle_extraction import EdgeDetectionMethod
from test_angle_extraction import test_angle_extraction


def main() -> None:
    image_directory = Path("../data/angle_extraction_test")
    save_dir = Path("../data/angle_extraction_test")
    # Run the angle extraction in parallel on every
    # png image, for both edge detection methods
    with Pool(multiprocessing.cpu_count()) as pool:
        res = [
            pool.apply_async(test_angle_extraction, (method, image, save_dir))
            for image in image_directory.glob("*.png")
            for method in EdgeDetectionMethod
        ]
        for r in res:
            r.get()


if __name__ == "__main__":
    main()
