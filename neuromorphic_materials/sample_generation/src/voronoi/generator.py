import os
from collections.abc import Callable

import numpy as np
import scipy

from ..rounding import int_round_half_away_from_zero
from .arg_parser import LocationUpdateMethod, VoronoiParser
from .cvd import CentroidalVoronoiDiagram
from .site import VoronoiSite
from .voronoi_generation_error import VoronoiGenerationError


class VoronoiGenerator:
    n_sites: int
    n_iterations: int
    point_generator: Callable[[tuple[int, ...]], np.ndarray]
    p_horizontal: float
    horizontal_beta: float
    _rng: np.random.Generator

    _n_cvds: int
    _site_update_method: Callable[[], []]
    _diagrams: np.ndarray | None
    _stacked_indices: np.ndarray | None
    _site_locations: np.ndarray | None
    _site_betas: np.ndarray | None

    def __init__(
        self,
        site_linear_density: float,
        p_horizontal: float,
        horizontal_beta: float,
        sample_size: int,
        n_iterations: int,
        site_update_method: LocationUpdateMethod,
        point_generator: Callable[[tuple[int, ...]], np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.n_sites = int_round_half_away_from_zero(site_linear_density * sample_size)
        self.n_iterations = n_iterations
        self.point_generator = point_generator
        self.p_horizontal = p_horizontal
        self.horizontal_beta = horizontal_beta
        if site_update_method == "centroid":
            self._site_update_method = self._update_sites_centroid
        # TODO: Fix hard core update method for use with ensemble generation
        # elif site_update_method == "hard_core":
        #     self._site_update_method = self._update_sites_hard_core
        else:
            raise LookupError(
                f"{site_update_method} is not a valid location update method name"
            )
        self._rng = rng
        indices = np.indices((sample_size, sample_size), dtype=np.float_)
        self._stacked_indices = np.moveaxis(indices, 0, 2)[
            ..., np.newaxis, np.newaxis
        ].repeat(self.n_sites, axis=3)

    @classmethod
    def from_arg_parser_args(cls, args: VoronoiParser):
        return cls(
            args.site_linear_density,
            args.p_horizontal,
            args.horizontal_beta,
            args.sample_size,
            args.n_iterations,
            args.site_update_method,
        )

    def _scaled_chebyshev_distances(self) -> np.ndarray:
        return (
            np.absolute(
                self._stacked_indices.repeat(self._n_cvds, axis=4)
                - self._site_locations
            )
            * self._site_betas
        ).max(axis=2)

    def _scaled_l2_distances(self) -> np.ndarray:
        return np.sqrt(
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
        new_locations = np.empty_like(self._site_locations)
        for cvd_idx in range(self._n_cvds):
            for idx in range(self.n_sites):
                region_indices = self._diagrams[..., cvd_idx] == idx
                if not region_indices.any():
                    raise VoronoiGenerationError(
                        "Cannot update site location with zero-size region"
                    )
                new_locations[:, idx, cvd_idx] = np.argwhere(region_indices).mean(
                    axis=0
                )
        self._site_locations = new_locations

    def _update_sites_hard_core(self) -> None:
        # TODO: Fix this function to work with ensemble generation
        new_site_locations = np.empty_like(self._site_locations)
        for idx, old_location in enumerate(self._site_locations):
            new_location = np.average((self._diagrams == idx).nonzero(), axis=1)
            masked_sites = np.ma.array(self._site_locations)
            masked_sites[idx] = np.ma.masked

            site_beta = self._site_betas[idx, 0]

            diff_oriented_sites = (
                self._site_betas[:, 0] < 1
                if site_beta > 1
                else self._site_betas[:, 0] > 1
            )
            # TODO: Allow moves that move away from differently oriented sites, but still are within the "hard core"
            dists_to_new_location = np.sqrt(
                ((masked_sites[diff_oriented_sites] - new_location) ** 2).sum(axis=1)
            )
            dists_to_old_location = np.sqrt(
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
        self._site_locations = (
            self._rng.random((2, self.n_sites, self._n_cvds))
            * self._stacked_indices.shape[0]
        )
        betas = self._rng.choice(
            (self.horizontal_beta, 1 / self.horizontal_beta),
            (self.n_sites, self._n_cvds),
            p=[self.p_horizontal, 1 - self.p_horizontal],
        )
        # Site betas has shape (2, n_sites, n_cvds)
        self._site_betas = np.stack((betas, 1 / betas), axis=0)

    def _extract_boundaries(self) -> np.ndarray:
        # TODO: See if sobel can be computed for all CVDs at once
        sample_size = self._stacked_indices.shape[0]
        boundaries = np.empty((self._n_cvds, sample_size, sample_size), dtype=np.uint8)
        for idx in range(self._n_cvds):
            sobel_x = scipy.ndimage.sobel(self._diagrams[..., idx], axis=0)
            sobel_y = scipy.ndimage.sobel(self._diagrams[..., idx], axis=1)
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
        for i in range(self.n_iterations):
            self._site_update_method()
            self._compute_diagrams()

            # This needs to stay commented!
            # save_iteration_image(self._diagrams, self._site_locations, iter=i, save_path=os.getcwd()+'/VoronoiTest/')


    def generate_cvds(self, count: int) -> list[CentroidalVoronoiDiagram]:
        self._generate(count)

        boundaries = self._extract_boundaries()

        return [
            CentroidalVoronoiDiagram(
                self._diagrams[..., idx], boundaries[idx], self._voronoi_sites(idx)
            )
            for idx in range(count)
        ]

    def generate_ensemble_boundaries(self, count) -> np.ndarray:
        self._generate(count)
        return self._extract_boundaries()


from matplotlib import pyplot as plt
def save_iteration_image(diagrams, site_locations, iter, save_path):
    fig, ax = plt.subplots()
    ax.imshow(diagrams, interpolation="none", cmap="coolwarm")
    ax.scatter(site_locations[1], site_locations[0], c="black", marker=".")
    ax.axis('off')
    fig.savefig("{:s}iter{:04d}.jpg".format(save_path, iter), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
