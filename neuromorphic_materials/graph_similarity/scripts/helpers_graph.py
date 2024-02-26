import networkx as nx
import numpy as np
import copy
import random

def remove_self_loops(graph):
    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(graph.nodes(data=True))
    temp_graph.add_edges_from(graph.edges(data=True))

    edges_to_be_removed = nx.selfloop_edges(graph)
    temp_graph.remove_edges_from(edges_to_be_removed)
    # print(list(edges_to_be_removed))
    return temp_graph

def merge_nodes_by_distance(g, epsilon=2.0, return_removed_nodes=False):

    # merged_graph = copy.deepcopy(g)
    merged_graph = nx.Graph()
    merged_graph.add_nodes_from(g.nodes(data=True))
    merged_graph.add_edges_from(g.edges(data=True))

    list_of_removed_nodes = []
    flag = 0
    while flag == 0:
        count = 0
        # print(count)
        for node1 in g.nodes():
            for node2 in g.nodes():
                if (node1 != node2) and (node1 not in list_of_removed_nodes) and (node2 not in list_of_removed_nodes): #(merged_graph.has_node(node1)) and (merged_graph.has_node(node2)) and
                    # if not isinstance(node1, str) or not isinstance(node2, str):
                        # print('Nodi sbagliati', node1, node2)
                        # raise TypeError(f"node1 and node2 must be of type str. node1: {type(node1)}, node2: {type(node2)}")
                    x1, y1 = map(int, str(node1).strip('()').split(', '))
                    x2, y2 = map(int, str(node2).strip('()').split(', '))
                    # print(x1, x2, type(x1), type(x2))
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if distance < epsilon:
                        count += 1
                        node_to_keep = random.choice([node1, node2])
                        node_to_remove = [node1, node2][0] if [node1, node2][1] == node_to_keep else [node1, node2][1]
                        nodes_to_connect = list(merged_graph.neighbors(node_to_remove))
                        list_of_removed_nodes.append(node_to_remove)

                        # The order of the following two lines matters! Sometimes in the neighbors of node to remove
                        # also appears the node to remove thus, if the order is reversed, the node to remove
                        # is reinserted in the graph and without its original dictionaries
                        merged_graph.add_edges_from([(node_to_keep, node) for node in nodes_to_connect])
                        merged_graph.remove_node(node_to_remove)

        if count == 0:
            flag = 1
    if return_removed_nodes:
        return merged_graph, list_of_removed_nodes
    else:
        return merged_graph

def connected_components(graph):
    """Return sub-graphs from largest to smaller"""
    sub_graphs = [
        graph.subgraph(c) for c in nx.connected_components(graph) if len(c) > 1
    ]
    sorted_sub_graphs = sorted(sub_graphs, key=len)
    return sorted_sub_graphs[::-1]
