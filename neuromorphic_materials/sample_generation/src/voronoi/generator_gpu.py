from collections.abc import Callable

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage

from ..rounding import int_round_half_away_from_zero
from .arg_parser import LocationUpdateMethod
from .cvd import CentroidalVoronoiDiagram
from .site import VoronoiSite
from .voronoi_generation_error import VoronoiGenerationError


class VoronoiGeneratorGPU:
    n_sites: int
    n_iterations: int
    point_generator: Callable[[tuple[int, ...]], cp.ndarray]
    p_horizontal: float
    horizontal_beta: float

    _n_cvds: int
    _site_update_method: Callable[[], []]
    _diagrams: cp.ndarray | None
    _stacked_indices: cp.ndarray | None
    _site_locations: cp.ndarray | None
    _site_betas: cp.ndarray | None

    def __init__(
        self,
        site_linear_density: float,
        p_horizontal: float,
        horizontal_beta: float,
        sample_size: int,
        n_iterations: int,
        site_update_method: LocationUpdateMethod,
        point_generator: Callable[[tuple[int, ...]], cp.ndarray] | None,
        rng_seed: int | None = None,
    ):
        self.n_sites = int_round_half_away_from_zero(site_linear_density * sample_size)
        self.n_iterations = n_iterations
        self.point_generator = point_generator
        self.p_horizontal = p_horizontal
        self.horizontal_beta = horizontal_beta
        if site_update_method == "centroid":
            self._site_update_method = self._update_sites_centroid
        # TODO: Fix hard core update method for use with CuPy
        # elif site_update_method == "hard_core":
        #     self._site_update_method = self._update_sites_hard_core
        else:
            raise LookupError(
                f"{site_update_method} is not a valid location update method name"
            )
        cp.random.seed(rng_seed)
        indices = cp.indices((sample_size, sample_size), dtype=cp.float_)
        self._stacked_indices = cp.moveaxis(indices, 0, 2)[
            ..., cp.newaxis, cp.newaxis
        ].repeat(self.n_sites, axis=3)

    def _scaled_chebyshev_distances(self) -> cp.ndarray:
        return (
            cp.absolute(
                self._stacked_indices.repeat(self._n_cvds, axis=4)
                - self._site_locations
            )
            * self._site_betas
        ).max(axis=2)

    def _scaled_l2_distances(self) -> cp.ndarray:
        return cp.sqrt(
            (
                (
                    self._stacked_indices.repeat(self._n_cvds, axis=4)
                    - self._site_locations
                )
                ** 2
                * self._site_betas**2
            ).sum(axis=2)
        )

    def _compute_diagrams(self) -> None:
        self._diagrams = self._scaled_chebyshev_distances().argmin(axis=2)

    def _update_sites_centroid(self) -> None:
        new_locations = cp.empty_like(self._site_locations)
        for cvd_idx in range(self._n_cvds):
            for idx in range(self.n_sites):
                region_indices = self._diagrams[..., cvd_idx] == idx
                if not region_indices.any():
                    raise VoronoiGenerationError(
                        "Cannot update site location with zero-size region"
                    )
                new_locations[:, idx, cvd_idx] = cp.argwhere(region_indices).mean(
                    axis=0
                )
        self._site_locations = new_locations

    def _update_sites_hard_core(self) -> None:
        # TODO: Fix masked array usage, which isn't available in CuPy
        new_site_locations = cp.empty_like(self._site_locations)
        for idx, old_location in enumerate(self._site_locations):
            new_location = cp.average((self._diagrams == idx).nonzero(), axis=1)
            masked_sites = cp.ma.array(self._site_locations)
            masked_sites[idx] = cp.ma.masked

            site_beta = self._site_betas[idx, 0]

            diff_oriented_sites = (
                self._site_betas[:, 0] < 1
                if site_beta > 1
                else self._site_betas[:, 0] > 1
            )
            # TODO: Allow moves that move away from differently oriented sites, but still are within the "hard core"
            dists_to_new_location = cp.sqrt(
                ((masked_sites[diff_oriented_sites] - new_location) ** 2).sum(axis=1)
            )
            dists_to_old_location = cp.sqrt(
                ((masked_sites[diff_oriented_sites] - old_location) ** 2).sum(axis=1)
            )

            # Disallow moves that move closer to other differently-oriented
            # sites and bring it within the hard core of such a site
            new_site_locations[idx] = (
                old_location
                if (
                    (dists_to_new_location < dists_to_old_location)
                    & (dists_to_new_location < 30)
                ).any()
                else new_location
            )

        return new_site_locations

    def _generate_sites(self) -> None:
        # Create Voronoi sites as shape (2, n_sites, n_cvds)
        # TODO: Use self.point_generator instead
        self._site_locations = (
            cp.random.random((2, self.n_sites, self._n_cvds))
            * self._stacked_indices.shape[0]
        )
        # TODO: Use cp.random.Generator when it supports choice()
        betas = cp.random.choice(
            (self.horizontal_beta, 1 / self.horizontal_beta),
            (self.n_sites, self._n_cvds),
            p=[self.p_horizontal, 1 - self.p_horizontal],
        )
        # Site betas has shape (2, n_sites, n_cvds)
        self._site_betas = cp.stack((betas, 1 / betas), axis=0)

    def _extract_boundaries(self) -> cp.ndarray:
        # TODO: See if sobel can be computed for all CVDs at once
        sample_size = self._stacked_indices.shape[0]
        boundaries = cp.empty((self._n_cvds, sample_size, sample_size), dtype=np.uint8)
        for idx in range(self._n_cvds):
            sobel_x = ndimage.sobel(self._diagrams[..., idx], axis=0)
            sobel_y = ndimage.sobel(self._diagrams[..., idx], axis=1)
            boundaries[idx] = np.where((sobel_x != 0) | (sobel_y != 0), 255, 0)

        # Compute sobel filter and move n_cvds axis to be first axis
        return boundaries

    def _voronoi_sites(self, cvd_idx: int) -> list[VoronoiSite]:
        return [
            VoronoiSite(
                self._site_locations[0, idx, cvd_idx].item(),
                self._site_locations[1, idx, cvd_idx].item(),
                self._site_betas[0, idx, cvd_idx].item(),
            )
            for idx in range(self.n_sites)
        ]

    def _generate(self, count: int) -> None:
        self._n_cvds = count
        self._generate_sites()
        self._compute_diagrams()
        for _ in range(self.n_iterations):
            self._site_update_method()
            self._compute_diagrams()

    def generate_cvds(self, count: int) -> list[CentroidalVoronoiDiagram]:
        self._generate(count)

        boundaries = self._extract_boundaries()

        return [
            CentroidalVoronoiDiagram(
                self._diagrams[..., idx].get(),
                boundaries[idx].get(),
                self._voronoi_sites(idx),
            )
            for idx in range(count)
        ]

    def generate_ensemble_boundaries(self, count) -> np.ndarray:
        self._generate(count)
        return self._extract_boundaries().get()
