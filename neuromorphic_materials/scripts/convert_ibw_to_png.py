#! /usr/bin/env python3

import math
from pathlib import Path

import cv2
import numpy as np
from neo import io
from tap import Tap
from varname import nameof


class ConvertIbwToPngParser(Tap):
    ibw_path: Path = None
    png_path: Path = None

    def configure(self) -> None:
        self.add_argument(nameof(self.ibw_path))


def load_array(name_file, index=3):
    r = io.IgorIO(name_file)
    array_data = r.read_analogsignal()
    notes = str(array_data.annotations["note"])
    # print(notes)
    # Size of the sample # For this sample is already 4microm
    one_over_micron = math.pow(10, 6)
    scan_size_micron = one_over_micron * float(
        (notes.split("b'ScanSize: "))[1].split("\\rFastScanSize")[0]
    )
    tipVoltage = float((notes.split("rSurfaceVoltage: "))[1].split("\\")[0])
    CurrentRetrace = array_data.rescale("nm").magnitude[:, :, index]
    num_px_per_micron = math.ceil(len(CurrentRetrace) / scan_size_micron)
    print("Name: ", name_file)
    print("ScanSize: ", scan_size_micron, "um")
    print("PixelSize: ", len(CurrentRetrace))
    print("Pixel_per_um: ", num_px_per_micron)
    print("SurfaceVoltage: ", tipVoltage)
    return CurrentRetrace, scan_size_micron, num_px_per_micron, tipVoltage


def main() -> None:
    args = ConvertIbwToPngParser().parse_args()
    array, *_ = load_array(args.ibw_path)
    norm_image = cv2.normalize(
        array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(
        str(
            args.png_path.resolve()
            if args.png_path is not None
            else args.ibw_path.with_suffix(".png").resolve()
        ),
        norm_image,
    )


if __name__ == "__main__":
    main()
