from collections.abc import Callable
from numbers import Number
from typing import TypeAlias, TypeVar

import numpy as np
from scipy.stats import expon, norm, uniform

from .point import Point

DistributionReturnType: TypeAlias = TypeVar("DistributionReturnType", bound=Number)

Distribution: TypeAlias = Callable[[], DistributionReturnType]


def length_distribution(distribution: Distribution) -> Distribution:
    return lambda: max(distribution(), 0)


def normal(mean: float, sd: float, rng: np.random.Generator) -> Callable[[], float]:
    return lambda: norm.rvs(size=1, loc=mean, scale=sd, random_state=rng).item(0)


def exponential(
    min_val: float, scale: float, rng: np.random.Generator
) -> Callable[[], float]:
    # TODO: Figure out correct exponential scale
    return lambda: expon.rvs(size=1, loc=min_val, scale=scale, random_state=rng).item(0)


def uniform_point(
    min_val: int, max_val: int, rng: np.random.Generator
) -> Callable[[], Point]:
    return lambda: Point(
        *uniform.rvs(size=2, loc=min_val, scale=max_val, random_state=rng).tolist()
    )
