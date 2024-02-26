import networkx as nx
import numpy as np
from community import community_louvain
from sklearn.preprocessing import PowerTransformer

from . import utils, visualize

def filter_graphs_based_on_maxdegree(graph_list, degree_threshold, return_percentage=False):
    """
        Filter a list of graphs based on a maximum degree threshold.

        Args:
        - graph_list (list): A list of NetworkX graphs to be filtered.
        - degree_threshold (int): The maximum allowed degree for nodes in a graph.
        - return_percentage (bool): If True, return the percentage of removed graphs.

        Returns:
        - filtered_graphs (list): A list of filtered graphs that meet the degree criterion.
        - percentage_removed (float): The percentage of removed graphs (if return_percentage is True).
    """
    filtered_graphs = []
    removed_count = 0
    total_graphs = len(graph_list)

    for graph in graph_list:
        max_degree = max(dict(graph.degree()).values())
        if max_degree < degree_threshold:
            g = nx.Graph()
            g.add_nodes_from(graph.nodes(data=True))
            g.add_edges_from(graph.edges(data=True))
            filtered_graphs.append(g)
        else:
            removed_count += 1

    if return_percentage:
        percentage_removed = (removed_count / total_graphs) * 100
        return filtered_graphs, percentage_removed

    return filtered_graphs

def merge_nodes_by_distance(g, epsilon, return_removed_nodes=False):
    import random

    merged_graph = nx.Graph()
    merged_graph.add_nodes_from(g.nodes(data=True))
    merged_graph.add_edges_from(g.edges(data=True))
    list_of_removed_nodes = []
    # list_nodes_to_keep = []

    flag = 0
    while flag == 0:
        count = 0

        for node1 in g.nodes():
            for node2 in g.nodes():
                if (node1 != node2) and (node1 not in list_of_removed_nodes) and (node2 not in list_of_removed_nodes): #(merged_graph.has_node(node1)) and (merged_graph.has_node(node2)) and
                    x1, y1 = map(int, str(node1).strip('()').split(', '))
                    x2, y2 = map(int, str(node2).strip('()').split(', '))
                    # x1, y1 = node1.strip('()').split(', ')
                    # x2, y2 = node2.strip('()').split(', ')
                    # x1 = int(x1)
                    # x2 = int(x2)
                    # y1 = int(y1)
                    # y2 = int(y2)

                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if distance < epsilon:
                        count += 1
                        node_to_keep = random.choice([node1, node2])
                        node_to_remove = [node1, node2][0] if [node1, node2][1] == node_to_keep else [node1, node2][1]
                        nodes_to_connect = list(merged_graph.neighbors(node_to_remove))

                        # # Check if nodes to connect are in
                        # in_second_list = [element for element in nodes_to_connect if element in list_of_removed_nodes]
                        # print(len(in_second_list), in_second_list)
                        #
                        # if node_to_remove in list_of_removed_nodes:
                        #     print(f"Node to remove {node_to_remove} is in removed nodes list.")
                        # if node_to_keep in list_of_removed_nodes:
                        #     print(f"Node to keep {node_to_keep} is in the removed nodes list.")

                        list_of_removed_nodes.append(node_to_remove)
                        # list_nodes_to_keep.append(node_to_keep)
                        # print(count, 'Rem:', node_to_remove, 'Keep:', node_to_keep)

                        # Remove all edges from "node1"
                        # merged_graph.remove_edges_from(merged_graph.edges(node_to_remove))

                        # The order of the following two lines matter! Sometimes in the neighbors of node to remove
                        # also appears the node to remove thus, if the order is reversed, the the node to remove
                        # is reinserted in the graph and without its original dictionaries
                        merged_graph.add_edges_from([(node_to_keep, node) for node in nodes_to_connect])
                        merged_graph.remove_node(node_to_remove)

                        # for n1 in merged_graph.nodes():
                        #     if n1 in list_of_removed_nodes:
                        #         print('n1', n1, 'in rem_list')
                        #         list(merged_graph.neighbors(n1))
                        #     start = merged_graph.nodes[n1]
                        #     a = [start["x"], start["y"]]
        if count == 0:
            flag =1
        # print('Merged nodes: ', count)
    # print('\n')
    if return_removed_nodes:
        return merged_graph, list_of_removed_nodes
    else:
        return merged_graph


def compute_degree_correlation_matrix(graphs, separate_matrices=True):
    """
    Compute the degree correlation matrix of a graph or a list of graphs.
    Each element represents the probability of finding
    two nodes of degree k and k' connected by a link.

    Parameters:
        graphs (networkx.Graph or list): The input graph or a list of graphs.
        separate_matrices (bool): If True, return a list of separate degree correlation matrices.

    Returns:
        degree_corr_matrix (numpy.ndarray or list): The degree correlation matrix or a list of matrices.
    """
    if isinstance(graphs, list):  # Handle a list of graphs
        max_degree = max(max(dict(G.degree()).values()) for G in graphs)
        if separate_matrices:
            degree_corr_matrices = []

            for graph in graphs:
                degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))
                for u, v in graph.edges():
                    degree_u = graph.degree[u]
                    degree_v = graph.degree[v]
                    degree_corr_matrix[degree_u][degree_v] += 1
                    degree_corr_matrix[degree_v][degree_u] += 1

                degree_corr_matrix /= (2 * graph.number_of_edges())
                degree_corr_matrices.append(degree_corr_matrix)

            return degree_corr_matrices

        else:
            degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))

            for graph in graphs:
                for u, v in graph.edges():
                    degree_u = graph.degree[u]
                    degree_v = graph.degree[v]
                    degree_corr_matrix[degree_u][degree_v] += 1
                    degree_corr_matrix[degree_v][degree_u] += 1

            degree_corr_matrix /= (2 * sum(G.number_of_edges() for G in graphs))

    else:  # Handle a single graph
        max_degree = max(dict(graphs.degree()).values())
        degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))

        for u, v in graphs.edges():
            degree_u = graphs.degree[u]
            degree_v = graphs.degree[v]
            degree_corr_matrix[degree_u][degree_v] += 1
            degree_corr_matrix[degree_v][degree_u] += 1

        degree_corr_matrix /= (2 * graphs.number_of_edges())

    return degree_corr_matrix



def degree_correlation_function(graphs, separate_matrices=False):
    '''
    Computes the conditional probability P(k'|k) = P(k,k')/P(k)
    and the computes the degree correlation function: knn(k) = Sum_k' k' * P(k'|k)

    :param (networkx.Graph or list): The input graph or a list of graphs.
    :param separate_matrices (bool): If True, compute separate matrices for each graph in the list.
                                    If False and if graphs is a list of graphs, it uses all the graphs in the list
                                    to compute the degree correlation matrix

    Returns:
        degree_range (list): List of node degrees.
        degree_correlation (list): knn(k) is the average degree of the neighbors of all degree-k nodes
    '''
    if isinstance(graphs, list):  # Handle a list of graphs
        if separate_matrices:
            degree_corr_matrices = compute_degree_correlation_matrix(graphs, separate_matrices=True)
            degree_range_list = []
            degree_correlation_list = []

            for graph, deg_corr_matx in zip(graphs, degree_corr_matrices):
                max_degree = max(dict(graph.degree()).values())
                degree_range = np.arange(max_degree + 1)
                row_sums = np.sum(deg_corr_matx, axis=1, keepdims=True)
                conditional_prob = deg_corr_matx / row_sums
                degree_corr_f = np.dot(conditional_prob, degree_range)
                degree_range_list.append(degree_range)
                degree_correlation_list.append(degree_corr_f)

            return degree_range_list, degree_correlation_list

        else:
            max_degree = max(max(dict(G.degree()).values()) for G in graphs)
            degree_corr_matrix = compute_degree_correlation_matrix(graphs, separate_matrices=False)
            row_sums = np.sum(degree_corr_matrix, axis=1, keepdims=True)
            conditional_prob = degree_corr_matrix / row_sums
            degree_range = np.arange(max_degree + 1)
            degree_corr_f = np.dot(conditional_prob, degree_range)

    else:  # Handle a single graph
        degree_corr_matrix = compute_degree_correlation_matrix(graphs)
        row_sums = np.sum(degree_corr_matrix, axis=1, keepdims=True)
        conditional_prob = degree_corr_matrix / row_sums
        max_degree = max(dict(graphs.degree()).values())
        degree_range = np.arange(max_degree + 1)
        degree_corr_f = np.dot(conditional_prob, degree_range)

    return degree_range, degree_corr_f




def plot_degree_correlation_matrix(degree_corr_matrix):
    """
    Plot the degree correlation matrix (normalized).

    Parameters:
        degree_corr_matrix (numpy.ndarray): The degree correlation matrix.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.imshow(degree_corr_matrix, origin='lower', cmap='viridis')
    # plt.colorbar(label='Number of Pairs')
    plt.colorbar(label='Probability of edge between node1 and node2\n'+r'$P(k, k\prime)$')
    plt.xlabel('Degree of Node 2')
    plt.ylabel('Degree of Node 1')
    plt.title('Degree Correlation Matrix')
    plt.show()


def degree_clustering_mean_std(degrees, clustering_coefficients):
    """
    Calculate the unique degree values, their corresponding mean clustering coefficients,
    and the standard deviations of the mean clustering coefficients.

    Parameters:
        degrees (numpy.ndarray): Array of degree values.
        clustering_coefficients (numpy.ndarray): Array of clustering coefficient values.

    Returns:
        unique_degrees (numpy.ndarray): Array of unique degree values.
        mean_clustering (numpy.ndarray): Array of mean clustering coefficients.
        std_clustering (numpy.ndarray): Array of standard deviations of mean clustering coefficients.
    """
    unique_degrees = np.unique(degrees)
    mean_clustering = np.zeros_like(unique_degrees, dtype=float)
    std_clustering = np.zeros_like(unique_degrees, dtype=float)

    for i, degree in enumerate(unique_degrees):
        matching_indices = np.where(degrees == degree)
        matching_clustering = clustering_coefficients[matching_indices]
        mean_clustering[i] = np.mean(matching_clustering)
        std_clustering[i] = np.std(matching_clustering)

    return unique_degrees, mean_clustering, std_clustering


def remove_degree_1_nodes(graph):
    """
    Iteratively removes nodes with degree 1 from a graph until no more nodes with degree 1 are left.

    Parameters:
        graph (networkx.Graph): Input graph.

    Returns:
        modified_graph (networkx.Graph): Graph with degree 1 nodes removed.
    """
    modified_graph = graph.copy()  # Create a copy of the input graph to modify

    while True:
        degree_1_nodes = [node for node, degree in dict(modified_graph.degree()).items() if degree == 1]
        if not degree_1_nodes:
            break

        modified_graph.remove_nodes_from(degree_1_nodes)

    return modified_graph




def set_new_weights(graph, new_weights):
    edge_list = [(n1, n2) for n1, n2, weight in list(graph.edges(data=True))]
    # edge_weight_list = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    edge_list_with_attr = [
        (edge[0], edge[1], {"weight": w}) for (edge, w) in zip(edge_list, new_weights)
    ]
    graph.add_edges_from(edge_list_with_attr)


def pruning_of_edges(graph, p, save_fold="./"):
    def eps_edges(edge_weight_list, p=0.25):
        pdf, cdf, bins = utils.empirical_pdf_and_cdf(edge_weight_list, bins=100)
        try:
            return bins[cdf < p][-1]
        except:
            return 0

    weights = [w["weight"] for u, v, w in graph.edges(data=True)]
    # Find weights to keep: weights > epsilon
    epsilon = eps_edges(p=p, edge_weight_list=weights)
    # plot cdf
    pdf, cdf, bins = utils.empirical_pdf_and_cdf(weights, bins=100)
    # plt.close()
    # plt.plot(bins, cdf)
    # plt.axvline(epsilon, c='red')
    # plt.xlabel('Edge weight')
    # plt.ylabel('CDF')
    # plt.savefig(save_fold + 'CDF_edges.png')
    # plt.close()
    #
    return [
        (u, v, {"weight": w["weight"]})
        for u, v, w in graph.edges(data=True)
        if w["weight"] > epsilon
    ]


def rescale_weights(graph, scale=(0.0001, 1), method="MinMax"):
    edge_list = [(n1, n2) for n1, n2, weight in list(graph.edges(data=True))]
    edge_weight_list = [
        weight["weight"] for n1, n2, weight in list(graph.edges(data=True))
    ]
    if method == "MinMax":
        edge_weight_list = utils.scale(edge_weight_list, scale)
    elif method == "box-cox":
        # print('Rescale method ', method)
        power = PowerTransformer(method="box-cox", standardize=True)
        data_trans = power.fit_transform(
            np.reshape(utils.scale(edge_weight_list, scale), newshape=(-1, 1))
        )
        edge_weight_list = utils.scale(data_trans.reshape(-1), scale)

    edge_list_with_attr = [
        (edge[0], edge[1], {"weight": w})
        for (edge, w) in zip(edge_list, edge_weight_list)
    ]
    graph.add_edges_from(edge_list_with_attr)


def connected_components(graph):
    """Return sub-graphs from largest to smaller"""
    sub_graphs = [
        graph.subgraph(c) for c in nx.connected_components(graph) if len(c) > 1
    ]
    sorted_sub_graphs = sorted(sub_graphs, key=len)
    return sorted_sub_graphs[::-1]


def relative_connection_density(graph, nodes):
    subG = graph.subgraph(nodes).copy()
    density = nx.density(subG)
    return density


def average_weighted_degree(graph, key_="weight"):
    """Average weighted degree of a graph"""
    edges_dict = graph.edges
    total = 0
    for node_adjacency_dict in edges_dict.values():
        total += sum(
            [adjacency.get(key_, 0) for adjacency in node_adjacency_dict.values()]
        )
    return total


def average_degree(graph):
    """Mean number of edges for a node in the network"""
    degrees = graph.degree()
    mean_num_of_edges = sum(dict(degrees).values()) / graph.number_of_nodes()
    return mean_num_of_edges


def filter_nodes_by_attr(graph, key_, key_value):
    """Returns the list of node indexes filtered by some value for the attribute key_"""
    return [
        idx for idx, (x, y) in enumerate(graph.nodes(data=True)) if y[key_] == key_value
    ]


def convert_to_networkx(node_list, edge_list_with_attr):
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list_with_attr)
    return graph


def centrality_plots(image, segments, graph, save_fold, save_name):
    # centrality= nx.get_node_attributes(G, 'information_centrality')
    centrality = nx.centrality.information_centrality(
        graph, weight=graph.edges(data=True)
    )
    _, bins = np.histogram(list(centrality.values()), 50)
    visualize.histogram(
        bins,
        list(centrality.values()),
        xlabel="Information Centrality",
        ylabel="# Nodes",
        title=f"Information Centrality {save_name}",
        save_fold=save_fold,
        save_name=f"InfoCentrality_hist_{save_name}",
    )
    nx.set_node_attributes(graph, centrality, "information_centrality")
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=graph,
        attr_color="information_centrality",
        save_fold=save_fold,
    )

    betweenness = nx.betweenness_centrality(graph)
    # betweenness = nx.get_node_attributes(G,'betweenness_centrality')
    _, bins = np.histogram(list(betweenness.values()), 50)
    visualize.histogram(
        bins,
        list(betweenness.values()),
        xlabel="Betweenness Centrality",
        ylabel="# Nodes",
        title=f"Betweenness Centrality {save_name}",
        save_fold=save_fold,
        save_name=f"betweenness_centrality_{save_name}",
    )
    nx.set_node_attributes(graph, centrality, "betweenness_centrality")
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=graph,
        attr_color="betweenness_centrality",
        save_fold=save_fold,
    )


def standard_analysis(image, segments, graph, save_fold, save_name):
    visualize.degree_analysis(
        graph, save_fold=save_fold, save_name=save_name, title=save_name
    )
    # visualize.draw_degree_colormap(G, save_fold=save_fold, save_name=save_name, title=save_name)
    visualize.adjaciency_matx(
        graph, save_fold=save_fold, save_name=save_name, title=save_name
    )
    visualize.plot_graph_over_image(graph, segments, image, save_fold=save_fold, lw=4)
    visualize.plot_superpix_image_point_color(
        image=image, segments=segments, G=graph, attr_color="lzw", save_fold=save_fold
    )
    visualize.plot_superpix_image_point_color(
        image=image,
        segments=segments,
        G=graph,
        attr_color="mean_intensity",
        save_fold=save_fold,
    )
    # visualize.plot_superpix_image_point_color(image=image, segments=segments, G=G, attr_color='information_centrality', save_fold=save_fold)
    # visualize.plot_superpix_image_point_color(image=image, segments=segments, G=G, attr_color='betweenness_centrality', save_fold=save_fold)

    weight = list(nx.get_edge_attributes(graph, "weight").values())
    visualize.histogram(
        bins=np.histogram(weight, 30)[1],
        values=weight,
        xlabel="Weight",
        ylabel="#",
        title=f"Weights Distribution {save_name}",
        save_fold=save_fold,
        save_name=f"weights_{save_name}",
    )


def community_detection(graph, save_fold, img_gray, segments):
    # compute the best partition
    resolution = 1.0
    partition = community_louvain.best_partition(
        graph, weight="weight", resolution=resolution
    )
    nx.set_node_attributes(graph, partition, "partition")
    visualize.plot_superpix_image_point_color(
        image=img_gray,
        segments=segments,
        G=graph,
        attr_color="partition",
        cbar_type="discrete",
        save_fold=save_fold,
    )
    return graph

def compute_clustering_coefficient_distribution(graph):
    """
    Compute the clustering coefficient distribution of nodes in a graph.

    Parameters:
        graph (networkx.Graph): Input graph.

    Returns:
        clustering_coefficients (numpy.ndarray): Array of clustering coefficients.
    """
    clustering_coefficients = np.array(list(nx.clustering(graph).values()))
    return clustering_coefficients

def entropy(L: np.array, beta_range: np.array):
    """
    This function computes Von Neumann spectral entropy over a range of beta values
    for a (batched) network with Laplacian L
    :math:`S(\\rho) = -\\mathrm{Tr}[\\rho \\log \\rho]`

    Parameters
    ----------
    L: np.array
        The (batched) n x n graph laplacian. If batched, the input dimension is [batch_size,n,n]

    beta_range: (iterable) list or numpy.array
        The range of beta

    Returns
    -------
    np.array
        The unnormalized Von Neumann entropy over the beta values and over all batches.
        Final dimension is [b,batch_size]. If 2D input, final dimension is [b]
        where b is the number of elements in the array beta_range

    Raises
    ------
    None
    """
    ndim = len(L.shape)
    lambd, Q = np.linalg.eigh(L)  # eigenvalues and eigenvectors of (batched) Laplacian
    if ndim == 3:
        batch_size = L.shape[0]
        entropy = np.zeros([batch_size, len(beta_range)])
        lrho = np.exp(-np.multiply.fun.outer(beta_range, lambd))
        Z = np.sum(lrho, axis=2)
        entropy = np.log(Z) + beta_range[:, None] * np.sum(lambd * lrho, axis=2) / Z
    elif ndim == 2:
        entropy = np.zeros_like(beta_range)
        for i, b in enumerate(beta_range):
            lrho = np.exp(-b * lambd)
            Z = lrho.sum()
            entropy[i] = np.log(np.abs(Z)) + b * (lambd * lrho).sum() / Z
    else:
        raise RuntimeError('Must provide a 2D or 3D array (as batched 2D arrays)')
    entropy[np.isnan(entropy)] = 0
    return entropy
