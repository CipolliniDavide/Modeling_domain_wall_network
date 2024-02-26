#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:04:37 2021

@author: hp
"""
# import simplejson as json
import networkx as nx


def read_nefi_graph(filename):
    import re

    import networkx as nx

    G = nx.read_multiline_adjlist(filename, delimiter="|")
    for u in G:
        G.nodes[u]["pos"] = [int(i) for i in re.split("\)|\(", u)[1].split(", ")]
    return G


def connected_components(G):
    # Return subgraphs from largest to smaller
    import networkx as nx

    subgraphs = [G.subgraph(c) for c in nx.connected_components(G) if len(c) > 1]
    sorted_subgraphs = sorted(subgraphs, key=len)
    return sorted_subgraphs[::-1]


def save_graph(G, fname):
    json.dump(
        dict(
            nodes=[[n, G.node[n]] for n in G.nodes()],
            edges=[[u, v, G.edge[u][v]] for u, v in G.edges()],
        ),
        open(fname, "w"),
        indent=2,
    )


def load_graph(fname):
    G = nx.Graph()
    d = json.load(open(fname))
    G.add_nodes_from(d["nodes"])
    G.add_edges_from(d["edges"])
    return G
