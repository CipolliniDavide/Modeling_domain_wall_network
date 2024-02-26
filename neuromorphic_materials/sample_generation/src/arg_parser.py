from tap import Tap

from .enum_action import EnumAction
from .sample_type import SampleType


class GenerateSampleParser(Tap):
    sample_type: SampleType = SampleType.DELAUNAY  # Sample type to generate

    def configure(self) -> None:
        self.add_argument("sample_type", action=EnumAction)
