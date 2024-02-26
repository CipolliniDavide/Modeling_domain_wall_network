from helpers.angle_extraction import EdgeDetectionMethod
from helpers.superpixels import SegmentationMethod
from tap import Tap
from typing_extensions import Literal

from .enum_action import EnumAction


class PrunedGraphArgParser(Tap):
    save_fold: str = "data/data_graph"
    sample_name: str = "JR38_far"
    segmentation_alg: SegmentationMethod = SegmentationMethod.SLIC
    edge_detection_alg: EdgeDetectionMethod = EdgeDetectionMethod.GAUSSIAN
    square_size: int = 32
    slic_n_segments: int = 200
    slic_sigma: float = 40.0
    alpha_def: Literal["fixed", "true"] = "fixed"
    polar_avg_window: int = 1
    prune_proba: float = 0.0
    max_distance: Literal["diagonal", "none"] = "diagonal"
    image_name: str = "JR38_far.png"
    df_load_path: str = "data/segmentation_data"
    show: bool = False
    save_polar_plots: bool = False

    def configure(self):
        self.add_argument("-s_f", "--save_fold")
        self.add_argument("-s_n", "--sample_name")
        self.add_argument(
            "-segmAlg",
            "--segmentation_alg",
            help="Image segmentation algorithm",
            action=EnumAction,
        )
        self.add_argument(
            "-edgeAlg",
            "--edge_detection_alg",
            help="Edge detection algorithm",
            action=EnumAction,
        )
        self.add_argument(
            "-ss", "--square_size", help="Size of each square in squares segmentation"
        )
        self.add_argument("-n_seg", "--slic_n_segments", help="Number of Superpixels")
        self.add_argument("-sigma", "--slic_sigma", help="Sigma for SLIC")
        self.add_argument(
            "-alpha_def",
            "--alpha_def",
            help=(
                'How alpha is computed: "fixed" for fixed distance used, or "true" for'
                " true distance between nodes"
            ),
        )
        self.add_argument(
            "-a",
            "--polar_avg_window",
            nargs="?",
            help=(
                "Size of the window used to smooth polar features, omit to disable"
                " smoothing"
            ),
        )
        self.add_argument("-p", "--prune_proba", help="Probability to erase weights")
        self.add_argument(
            "-max_d",
            "--max_distance",
            help="Connect only nodes closer than max_distance",
        )
        self.add_argument("-i", "--image_name")
        self.add_argument("-l", "--df_load_path")
        self.add_argument("-s", "--show", help="Show Plot")
