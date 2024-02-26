#! /usr/bin/env python3

import argparse
from os import pardir
from os.path import abspath, join
from pathlib import Path

import numpy as np
from arg_parsers.enum_action import EnumAction
from helpers import utils
from helpers.angle_extraction import EdgeDetectionMethod
from helpers.superpixels import (
    SegmentationMethod,
    SuperpixelFeatures,
    get_segm_path_part,
)
from skimage.segmentation import slic


def segment_squares(image, square_size):
    img_x = image.shape[0]
    img_y = image.shape[1]
    n_x = img_x // square_size
    n_y = img_y // square_size
    overflow_x = 1 if img_x % square_size else 0
    overflow_y = 1 if img_y % square_size else 0

    segments = np.empty((img_x, img_y), dtype=np.uint)
    for y in range(n_y):
        for x in range(n_x):
            segments[
                y * square_size : (y + 1) * square_size,
                x * square_size : (x + 1) * square_size,
            ] = (1 + y * (n_x + overflow_x) + x)
        if overflow_x:
            segments[
                y * square_size : (y + 1) * square_size, n_x * square_size : img_x
            ] = (1 + y * (n_x + 1) + n_x)
    if overflow_y:
        for x in range(n_x):
            segments[
                n_y * square_size : img_y, x * square_size : (x + 1) * square_size
            ] = (1 + n_y * (n_x + overflow_x) + x)

    return segments


def main():
    # define parser & its arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s_f", "--save_fold", default="data/segmentation_data")
    parser.add_argument(
        "-segmAlg",
        "--segmentation_alg",
        default=SegmentationMethod.SLIC,
        help="Image segmentation algorithm",
        type=SegmentationMethod,
        action=EnumAction,
    )
    parser.add_argument(
        "-edgeAlg",
        "--edge_detection_alg",
        default=EdgeDetectionMethod.SOBEL,
        help="Edge detection algorithm",
        type=EdgeDetectionMethod,
        action=EnumAction,
    )
    parser.add_argument(
        "-ss",
        "--square_size",
        default=32,
        help="Square size for squares segmentation",
        type=int,
    )
    parser.add_argument(
        "-nSeg",
        "--slic_n_segments",
        default=200,
        help="Number of Superpixels",
        type=int,
    )
    parser.add_argument(
        "-sigma", "--slic_sigma", default=40, help="Sigma for slic", type=float
    )
    parser.add_argument("-s", "--show", default=False, nargs="?", help="Show Plot")
    parser.add_argument("-i", "--img_path", default="JR38_far.png")
    # parse
    args = parser.parse_args()

    root_dir = abspath(join(".", pardir))
    load_path = Path(root_dir) / args.img_path

    segm_path_part = get_segm_path_part(args)

    if segm_path_part is None:
        print("Segmentation algorithm is invalid, quitting.")
        exit(1)

    # Save path
    save_fold = join(
        root_dir,
        args.save_fold,
        load_path.stem,
        segm_path_part,
        f"edgeDetectAlg_{args.edge_detection_alg.value}",
    )
    utils.ensure_dir(save_fold)

    print(f"Save folder: {save_fold}")

    # load image
    image_gray, image_rgb = utils.load_image(load_path)

    # Build graph with SLIC
    # segments = np.load(save_fold_temp + 'segments.npy')

    segments = None
    if args.segmentation_alg == SegmentationMethod.SLIC:
        segments = slic(
            image_rgb,
            n_segments=args.slic_n_segments,
            sigma=args.slic_sigma,
            start_label=1,
            convert2lab=True,
        )
    elif args.segmentation_alg == SegmentationMethod.SQUARES:
        segments = segment_squares(image_rgb, args.square_size)

    if segments is None:
        print("Segmentation algorithm is invalid, quitting.")
        exit(1)

    # segments = slic(image_rgb, compactness=30, n_segments=n_segments, start_label=1)
    # Set bin number to 361 for gaussian rotate image because it always produces 360 bins
    bin_num = (
        361
        if args.edge_detection_alg == EdgeDetectionMethod.GAUSSIAN_ROTATE_IMG
        else 121
    )
    SuperpixelFeatures(
        image=image_gray,
        segments=segments,
        edge_alg=args.edge_detection_alg,
        bin_num=bin_num,
    ).create_multiprocess(save_name="df", save_fold=save_fold)


if __name__ == "__main__":
    main()
