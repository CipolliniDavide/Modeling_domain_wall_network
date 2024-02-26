import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self, TypeAlias, TypedDict, TypeVar

from ..point import SizeTuple
from .image import SampleImage


# TODO: Allow parametrizing this class based on the type of sample, so methods like
#  from_dict and to_dict can use the correct typing. Looking for something like
#  SampleDict<SampleType>, i.e. a sample dictionary for the specific sample type.
#  If that is not possible, define the methods without typing in this base class,
#  and define the specific types in the subclasses.
class BaseSampleDict(TypedDict):
    sample_type: str
    sample_size: SizeTuple


class BaseSample(ABC):
    sample_type: ClassVar[str] = "base_sample"

    def __init__(self, sample_size: SizeTuple) -> None:
        self._sample_size: SizeTuple = sample_size

    @classmethod
    @abstractmethod
    def from_dict(cls, sample_dict: BaseSampleDict) -> Self:
        pass

    @abstractmethod
    def to_dict(self) -> BaseSampleDict:
        pass

    @classmethod
    def from_json_string(cls, json_string: str) -> Self:
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_json_file(cls, file_path: Path) -> Self:
        with file_path.open("r") as json_file:
            return cls.from_dict(json.load(json_file))

    @abstractmethod
    def draw(self, image: SampleImage) -> None:
        pass

    def save(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as out_file:
            json.dump(self.to_dict(), out_file)

    def generate_image(self) -> SampleImage:
        image = SampleImage(self._sample_size)
        self.draw(image)
        return image


TSample: TypeAlias = TypeVar("TSample", bound=BaseSample)
