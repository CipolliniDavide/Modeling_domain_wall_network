import math


def int_round_half_away_from_zero(number: float) -> int:
    return int(math.copysign(math.ceil(abs(number) - 0.5), number))
