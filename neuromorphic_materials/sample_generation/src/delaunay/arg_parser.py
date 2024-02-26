from ..base_sample import BaseSampleParser


class DelaunaySampleParser(BaseSampleParser):
    patch_size: int = 32  # Size of patches in pixels (squared)
    density_high: float = 0.01  # Density for high density patches
    density_low: float = 0.0  # Density for low density patches
    p_density_high: float = 0.3  # Probability of a high density patch
