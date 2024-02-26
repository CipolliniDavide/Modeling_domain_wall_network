#!/usr/bin/env python3

import random
from statistics import mean

import networkx as nx


class RandomGraph:
    def __init__(self, nodes, mean_degree, mean_weight):
        self.nodes = nodes
        self.meanDegree = mean_degree
        self.meanWeight = mean_weight
        self.edges = 0
        self.weight = 0
        self.graph = [[0 for i in range(0, self.nodes)] for j in range(0, self.nodes)]
        self.positions = [
            (i, j) for i in range(0, self.nodes) for j in range(0, self.nodes) if i < j
        ]
        random.shuffle(self.positions)

    def avg_degree(self):
        return (self.edges * 2.0) / self.nodes

    def avg_weight(self):
        return self.weight / self.edges

    def add_edge(self, i, j, weight=1):
        if self.graph[i][j] == 0 and self.graph[j][i] == 0:
            self.graph[i][j] = weight
            self.graph[j][i] = weight
            self.edges += 1
            self.weight += weight
            self.positions.remove((i, j))

    def add_weight(self, i, j, add=1):
        if self.graph[i][j] > 0:
            self.graph[i][j] += add
            self.graph[j][i] += add
            self.weight += add

    def remove_edge(self, i, j):
        self.graph[i][j] = 0
        self.graph[j][i] = 0

    def get_edges(self):
        return [
            (i, j, self.graph[i][j])
            for i in range(0, self.nodes)
            for j in range(0, self.nodes)
            if i < j and self.graph[i][j] > 0
        ]

    def get_matrix(self):
        return self.graph

    def get_node(self, node):
        return [
            (j, self.graph[node][j])
            for j in range(0, self.nodes)
            if self.graph[node][j] > 0
        ]

    def get_nodes(self):
        return [(i, self.get_node(i)) for i in range(0, self.nodes)]

    def create_graph(self):
        # First connect even nodes with odd nodes
        for i in range(0, self.nodes, 2):
            if self.avg_degree() >= self.meanDegree:
                break
            if i + 1 < self.nodes:
                self.add_edge(i, i + 1)
        # Now connect odd nodes with even nodes (make chain)
        for i in range(1, self.nodes, 2):
            if self.avg_degree() >= self.meanDegree:
                break
            if i + 1 < self.nodes:
                self.add_edge(i, i + 1)
        if self.avg_degree() < self.meanDegree:
            # Close the chain
            self.add_edge(0, self.nodes - 1)
        # At this point we should start edges randomly until we have reach the average degree
        while len(self.positions) > 0:
            if self.avg_degree() >= self.meanDegree:
                break
            (i, j) = self.positions[0]
            self.add_edge(i, j)
        # Now we have to increase weights until we reach the needed average
        existing_edges = self.get_edges()
        while self.avg_weight() < self.meanWeight:
            (i, j, weight) = random.choice(existing_edges)
            self.add_weight(i, j)


# from random_graph_set_mean_degree import RandomGraph
# graph = RandomGraph(24, 5, mean_degree)
# graph.createGraph()
# print("All graph edges with weights, list of (node1, node2, weight) tuples\n", graph.getEdges())
# print("Nodes connected to node 1, with weights, list of (node, weigh) tuples\n", graph.getNode(1))
# print("Complete node info, list of getNode(i) values for all nodes\n", graph.getNodes())
# print("Matrix representation, element a[i][j] has the weight of connecting edge, 0 otherwise\n", graph.getMatrix())
# print("Average degree of node\n", graph.avgDegree())
# print("Average edge weight\n", graph.avgWeight())


# ---


class MyGraph(nx.Graph):
    def __init__(
        self, num_nodes, target_deg, weighted=False, target_wght=3, max_wght=5
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.target_deg = target_deg
        self.target_wght = target_wght
        self.max_wght = max_wght
        self.add_nodes_from(range(self.num_nodes))
        while self.avg_deg() < self.target_deg:
            n1, n2 = random.sample(self.nodes(), 2)
            self.add_edge(n1, n2, weight=1)
        if weighted:
            while self.avg_wght() < self.target_wght:
                n1, n2 = random.choice(list(self.edges()))
                if self[n1][n2]["weight"] < self.max_wght:
                    self[n1][n2]["weight"] += 1

    def avg_deg(self):
        return self.number_of_edges() * 2 / self.num_nodes

    def avg_wght(self):
        wghts = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                try:
                    wghts.append(self[i][j]["weight"])
                except KeyError:
                    pass
        return mean(wghts)
