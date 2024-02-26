#!/usr/bin/env python
"""
================
Degree histogram
================

Draw degree histogram with matplotlib.
Random graph shown as inset
"""


def plot_degree_histogram(G, draw_graph=False, save_path=None):
    import collections

    import matplotlib.pyplot as plt
    import networkx as nx

    # G = nx.gnp_random_graph(100, 0.02)

    degree_sequence = sorted(
        [d for n, d in G.degree()], reverse=True
    )  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")
    # print(cnt)
    # print(deg)
    plt.title("%s. Degree Histogram" % save_path.split("/")[1])
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(deg)
    if draw_graph:
        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        A = (G.subgraph(c) for c in nx.connected_components(G))
        A = list(A)[0]
        # Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
        Gcc = sorted(A, key=len, reverse=True)[0]
        pos = nx.spring_layout(G)
        plt.axis("off")
        nx.draw_networkx_nodes(G, pos, node_size=20)
        nx.draw_networkx_edges(G, pos, alpha=0.4)
    if save_path:
        plt.savefig(save_path + "degree_dist.png")
    plt.close()


def plot_degree_dist(G, save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    if save_path:
        plt.savefig(save_path + "degree_dist.png")
    plt.close()


def plot_degree_freq(G, m=0, save_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.title("%s. Degree frequency" % save_path.split("/")[1])
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(degrees[m:], degree_freq[m:], "go-")

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path + "degree_freq.png")
    plt.close()
