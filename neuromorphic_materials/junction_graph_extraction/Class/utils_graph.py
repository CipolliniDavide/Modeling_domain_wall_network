import networkx as nx
import numpy as np


def rescale_weights(G, scale=(0.0001, 1)):
    from .utils import utils

    edge_list = [(n1, n2) for n1, n2, weight in list(G.edges(data=True))]
    edge_weight_list = [weight["weight"] for n1, n2, weight in list(G.edges(data=True))]
    edge_weight_list = utils.scale(edge_weight_list, scale)
    edge_list_with_attr = [
        (edge[0], edge[1], {"weight": w})
        for (edge, w) in zip(edge_list, edge_weight_list)
    ]
    G.add_edges_from(edge_list_with_attr)


def connected_components(G):
    # Return subgraphs from largest to smaller
    import networkx as nx

    subgraphs = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
    sorted_subgraphs = sorted(subgraphs, key=len)
    return sorted_subgraphs[::-1]


def relative_connection_density(G, nodes):
    subG = G.subgraph(nodes).copy()
    density = nx.density(subG)
    return density


def average_weighted_degree(G, key_="weight"):
    """Average weighted degree of a graph"""
    edgesdict = G.edges
    total = 0
    for node_adjacency_dict in edgesdict.values():
        total += sum(
            [adjacency.get(key_, 0) for adjacency in node_adjacency_dict.values()]
        )
    return total


def average_degree(G):
    "Mean number of edges for a node in the network"
    degrees = G.degree()
    mean_num_of_edges = sum(dict(degrees).values()) / G.number_of_nodes()
    return mean_num_of_edges


def filter_nodes_by_attr(G, key_, key_value):
    "Returns the list of node indexs filtered by some value for the attribute key_"
    return [
        idx for idx, (x, y) in enumerate(G.nodes(data=True)) if y[key_] == key_value
    ]
