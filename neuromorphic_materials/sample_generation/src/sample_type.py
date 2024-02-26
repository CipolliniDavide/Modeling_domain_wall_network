from enum import StrEnum

from .base_sample import TSample, TSampleGenerator, TSampleParser
from .bfo_like import BFOLikeSample, BFOLikeSampleGenerator, BFOLikeSampleParser
from .crystal import CrystalSample, CrystalSampleGenerator, CrystalSampleParser
from .delaunay import DelaunaySample, DelaunaySampleGenerator, DelaunaySampleParser


class SampleType(StrEnum):
    BFO_LIKE = BFOLikeSample.sample_type
    DELAUNAY = DelaunaySample.sample_type
    CRYSTAL = CrystalSample.sample_type

    def arg_parser(self) -> TSampleParser:
        return _ARG_PARSER_TYPE_MAP[self]()

    def sample_type(self) -> type[TSample]:
        return _SAMPLE_TYPE_MAP[self]

    def generator_type(self) -> type[TSampleGenerator]:
        return _GENERATOR_TYPE_MAP[self]


_ARG_PARSER_TYPE_MAP: dict[SampleType, type[TSampleParser]] = {
    SampleType.BFO_LIKE: BFOLikeSampleParser,
    SampleType.DELAUNAY: DelaunaySampleParser,
    SampleType.CRYSTAL: CrystalSampleParser,
}

_SAMPLE_TYPE_MAP: dict[SampleType, type[TSample]] = {
    SampleType.BFO_LIKE: BFOLikeSample,
    SampleType.DELAUNAY: DelaunaySample,
    SampleType.CRYSTAL: CrystalSample,
}

_GENERATOR_TYPE_MAP: dict[SampleType, type[TSampleGenerator]] = {
    SampleType.BFO_LIKE: BFOLikeSampleGenerator,
    SampleType.DELAUNAY: DelaunaySampleGenerator,
    SampleType.CRYSTAL: CrystalSampleGenerator,
}
