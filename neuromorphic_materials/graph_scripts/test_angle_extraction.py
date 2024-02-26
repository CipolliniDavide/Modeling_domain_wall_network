#! /usr/bin/env python3
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from arg_parsers import TestAngleExtractionArgParser
from helpers import utils
from helpers.angle_extraction import (
    EdgeDetectionMethod,
    extract_angles_gaussian_45_deg_eval,
    extract_pixel_angles_gaussian,
    extract_pixel_angles_sobel,
)
from helpers.polar_plot import polar_plot
from helpers.superpixels import SuperpixelGraph
from helpers.utils import load_image
from matplotlib import pyplot as plt


def get_polar_feats_pdf(polar_features: np.ndarray) -> np.ndarray:
    # Using 361 bins will lead to 360 individual values, for finer results
    # The normal script uses 121 bins, for 120 values (3 degrees per bin)
    return utils.empirical_pdf_and_cdf(
        polar_features, bins=np.linspace(0, 360, num=121, dtype=float)
    )[0]


def visualise_45_deg_angles(
    angles: np.ndarray, indices: np.ndarray, window: float = 1, save_name: Path = None
) -> None:
    img = np.full((512, 512), -255)
    img[indices] = angles
    masked = np.zeros((512, 512))
    masked[
        (img >= -135 - window) & (img <= -135 + window)
        | (img >= -45 - window) & (img <= -42 + window)
        | (img >= 45 - window) & (img <= 45 + window)
        | (img >= 135 - window) & (img <= 135 + window)
    ] = 255
    if save_name is not None:
        save_name.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(save_name, masked, cmap="inferno")
    else:
        plt.imshow(masked)
        plt.show()


def process_and_plot(polar_features_pdf: np.ndarray, save_dir: Path) -> None:
    for a in [1, 3, 5, 7, 9]:
        cleaned_feats_pdf = SuperpixelGraph.clean_superpixel_polar_feats(
            polar_features_pdf, np.ones(a) / a if a != 1 else None, False
        )
        polar_plot(
            cleaned_feats_pdf,
            save_filename=f"a_{a}",
            show=(save_dir is None),
            save_dir=save_dir,
        )


def test_gaussian(image: np.ndarray, save_dir: Path) -> None:
    for sigma in [1, 2, 3, 5]:
        angles, indices = extract_pixel_angles_gaussian(image, 5, sigma)
        sigma_dir = save_dir / f"gaussian_sigma_{sigma}"
        visualise_45_deg_angles(angles, indices, 2, sigma_dir / "45_deg_angles.png")
        process_and_plot(get_polar_feats_pdf(angles), sigma_dir)


def test_gaussian_rotate_img(image: np.ndarray, save_dir: Path) -> None:
    for sigma in [1, 2, 3, 5]:
        angles = extract_angles_gaussian_45_deg_eval(image, 5, sigma, 1.5)
        sigma_dir = save_dir / f"gaussian_rot_img_sigma_{sigma}"
        process_and_plot(angles, sigma_dir)


def test_sobel(image: np.ndarray, save_dir: Path) -> None:
    angles, indices = extract_pixel_angles_sobel(image)
    sobel_dir = save_dir / "sobel"
    visualise_45_deg_angles(angles, indices, 2, sobel_dir / "45_deg_angles.png")
    process_and_plot(get_polar_feats_pdf(angles), sobel_dir)


def test_angle_extraction(
    edge_detection_alg: EdgeDetectionMethod, image_path: Path, save_dir: Path
) -> None:
    if not image_path.exists():
        raise FileNotFoundError("Input image does not exist")

    # Map edge detection methods to testing functions
    edge_detect_map: Dict[EdgeDetectionMethod, Callable[[np.ndarray, Path], None]] = {
        EdgeDetectionMethod.SOBEL: test_sobel,
        EdgeDetectionMethod.GAUSSIAN: test_gaussian,
        EdgeDetectionMethod.GAUSSIAN_ROTATE_IMG: test_gaussian_rotate_img,
    }

    # Load the image
    img_gray, _ = load_image(image_path)

    if edge_detection_alg not in edge_detect_map:
        raise KeyError("Invalid edge detection algorithm")
    edge_detect_map[edge_detection_alg](img_gray, save_dir / image_path.stem)


def main() -> None:
    arg_parser = TestAngleExtractionArgParser()
    args = arg_parser.parse_args()

    test_angle_extraction(args.edge_detection_alg, args.image_path, args.save_dir)


if __name__ == "__main__":
    main()
