from pathlib import Path

from helpers.angle_extraction import EdgeDetectionMethod
from tap import Tap


class TestAngleExtractionArgParser(Tap):
    edge_detection_alg: EdgeDetectionMethod = (  # Edge detection for angle extraction
        EdgeDetectionMethod.GAUSSIAN_ROTATE_IMG
    )
    image_path: Path = Path("../JR38_far.png")  # Path to the input image
    save_dir: Path = Path(  # Folder to save the resulting polar histograms to
        "../data/angle_extraction_test"
    )

    def configure(self):
        self.add_argument("-e", "--edge_detection_alg")
        self.add_argument("-i", "--image_path")
        self.add_argument("-s", "--save_dir")
