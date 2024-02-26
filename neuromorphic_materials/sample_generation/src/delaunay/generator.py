import math

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from ..base_sample import BaseSampleGenerator
from ..point import ImagePoint, Point
from . import DelaunaySampleParser
from .sample import DelaunayEdge, DelaunayNode, DelaunaySample


class DelaunaySampleGenerator(BaseSampleGenerator):
    def __init__(self, args: DelaunaySampleParser) -> None:
        super().__init__(args)
        self._args = args

    def _generate_random_vertex(
        self, patch_start: ImagePoint
    ) -> npt.NDArray[np.float_]:
        return np.floor(self._args.patch_size * self._rng.random((2,))) + patch_start

    def _generate_patch_vertices(
        self, patch_start: ImagePoint
    ) -> list[npt.NDArray[np.float_]]:
        # TODO: ensure points are not generated out of bounds if the image
        #  does not divide exactly into patch_size * patch_size squares
        density = (
            self._args.density_high
            if self._rng.random() < self._args.p_density_high
            else self._args.density_low
        )

        vertices: list[npt.NDArray[np.float_]] = []

        for _ in range(math.floor(density * self._args.patch_size**2)):
            # Ensure only unique vertices are generated and added to the patch
            vertex = self._generate_random_vertex(patch_start)
            while vertex in np.asarray(vertices):
                vertex = self._generate_random_vertex(patch_start)
            vertices.append(vertex)

        return vertices

    def generate_sample(self) -> DelaunaySample:
        # Generate vertices in each patch
        vertices: list[npt.NDArray[np.float_]] = []
        for x_patch in range(0, self._args.sample_size, self._args.patch_size):
            for y_patch in range(0, self._args.sample_size, self._args.patch_size):
                vertices += self._generate_patch_vertices((x_patch, y_patch))

        # Perform Delaunay triangulation
        delaunay = Delaunay(np.array(vertices), qhull_options="QJ")

        # Combine triangle points as edges
        # Produces an array of shape [ [ [x1 y1], [x2 y2] ],, [ ... ],, ... ]
        edges: list[tuple[npt.NDArray[np.float_]]] = []
        for points in delaunay.points[delaunay.simplices]:
            # Sort edges before taking unique to remove inverse edges, as edges are not
            # directional. So: (p1, p2) == (p2, p1)
            points: list[npt.NDArray[np.float_]] = sorted(
                points, key=lambda p: (p[0], p[1])
            )
            # Only add edges that are not already in the edge array
            edges += [
                edge
                for edge in (
                    (points[0], points[1]),
                    (points[0], points[2]),
                    (points[1], points[2]),
                )
                if len(edges) == 0
                or not np.any(np.all(np.isclose(edge, edges), axis=(1, 2)))
            ]

        sample = DelaunaySample((self._args.sample_size, self._args.sample_size))
        # Add all nodes to the sample
        for vertex in vertices:
            sample.add_node(DelaunayNode(Point.from_np_arr(vertex)))

        # Add all unique edges to the sample
        for edge in edges:
            sample.add_edge(
                DelaunayEdge(Point.from_np_arr(edge[0]), Point.from_np_arr(edge[1]))
            )

        return sample
