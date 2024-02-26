#! /usr/bin/env python3

from pathlib import Path

import cv2
from tap import Tap


class CropSampleParser(Tap):
    input_file: Path
    output_dir: Path
    crop_size: int = 128

    def configure(self) -> None:
        self.add_argument("input_file")
        self.add_argument("output_dir")


def main() -> None:
    parser = CropSampleParser()
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"The input file {args.input_file} does not exist")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = cv2.imread(str(args.input_file.resolve()), cv2.IMREAD_GRAYSCALE)
    for y in range(0, sample.shape[0], args.crop_size):
        for x in range(0, sample.shape[1], args.crop_size):
            cv2.imwrite(
                str(
                    (args.output_dir / f"{args.input_file.stem}_{x}_{y}")
                    .with_suffix(".png")
                    .resolve()
                ),
                sample[y : y + args.crop_size, x : x + args.crop_size],
            )


if __name__ == "__main__":
    main()
