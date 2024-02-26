from abc import ABC, abstractmethod
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.random import Generator

from . import TSampleParser
from .sample import TSample


class BaseSampleGenerator(ABC):
    def __init__(self, args: TSampleParser) -> None:
        self._rng: Generator = np.random.default_rng()
        # TODO: Allow correct type inference of sample parser args in
        #  sample subclasses, without explicitly setting self._args
        self._args: TSampleParser = args

    @abstractmethod
    def generate_sample(self) -> TSample:
        pass


TSampleGenerator: TypeAlias = TypeVar("TSampleGenerator", bound=BaseSampleGenerator)
