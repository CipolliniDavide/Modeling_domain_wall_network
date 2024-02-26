from typing import ClassVar, Self, TypedDict

import networkx as nx

from ..base_sample import BaseSample, BaseSampleDict, SampleImage, SampleLine
from ..point import Edge, Point, PointTuple, SizeTuple


class DelaunayNodeDict(TypedDict):
    coordinates: PointTuple


class DelaunayNode:
    def __init__(self, coordinates: Point) -> None:
        self._coordinates = coordinates

    def to_dict(self) -> DelaunayNodeDict:
        return DelaunayNodeDict(coordinates=self._coordinates.to_tuple())

    @classmethod
    def from_dict(cls, data: DelaunayNodeDict) -> Self:
        return cls(Point.from_tuple(data["coordinates"]))

    def to_tuple(self) -> PointTuple:
        return self._coordinates.to_tuple()


class DelaunayEdgeDict(TypedDict):
    start_point: PointTuple
    end_point: PointTuple


class DelaunayEdge:
    def __init__(self, start_point: Point, end_point: Point) -> None:
        self._start_point = start_point
        self._end_point = end_point

    def to_dict(self) -> DelaunayEdgeDict:
        return DelaunayEdgeDict(
            start_point=self._start_point.to_tuple(),
            end_point=self._end_point.to_tuple(),
        )

    @classmethod
    def from_dict(cls, data: DelaunayEdgeDict) -> Self:
        return cls(
            Point.from_tuple(data["start_point"]), Point.from_tuple(data["end_point"])
        )

    def to_sample_line(self) -> SampleLine:
        return SampleLine.between_points(
            self._start_point, self._end_point, thickness=1
        )

    def to_tuple(self) -> Edge:
        return self._start_point, self._end_point


class DelaunaySampleDict(BaseSampleDict):
    nodes: list[DelaunayNodeDict]
    edges: list[DelaunayEdgeDict]


class DelaunaySample(BaseSample):
    sample_type: ClassVar[str] = "delaunay"

    def __init__(self, sample_size: SizeTuple) -> None:
        super().__init__(sample_size)
        self._nodes: list[DelaunayNode] = []
        self._edges: list[DelaunayEdge] = []

    @classmethod
    def from_dict(cls, sample_dict: DelaunaySampleDict) -> Self:
        sample = cls(sample_dict["sample_size"])
        sample._nodes = [DelaunayNode.from_dict(node) for node in sample_dict["nodes"]]
        sample._edges = [DelaunayEdge.from_dict(edge) for edge in sample_dict["edges"]]
        return sample

    def to_dict(self) -> DelaunaySampleDict:
        return DelaunaySampleDict(
            sample_type=DelaunaySample.sample_type,
            sample_size=self._sample_size,
            nodes=[node.to_dict() for node in self._nodes],
            edges=[edge.to_dict() for edge in self._edges],
        )

    def draw(self, image: SampleImage) -> None:
        for edge in self._edges:
            image.draw_line(edge.to_sample_line())

    def generate_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(
            (node.to_tuple(), {"coords": node.to_tuple()}) for node in self._nodes
        )
        graph.add_edges_from(edge.to_tuple() for edge in self._edges)
        return graph

    def add_node(self, node: DelaunayNode) -> None:
        self._nodes.append(node)

    def add_edge(self, edge: DelaunayEdge) -> None:
        self._edges.append(edge)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)
