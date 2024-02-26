#! /usr/bin/env python3

import copy
import glob
import json
import os
from os.path import abspath, join

import easydict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from arg_parsers import CreateDatasetArgParser, PrunedGraphArgParser
from helpers import utils, visualize
from helpers.graph import rescale_weights
from helpers.visual import create_colorbar
from networkx.algorithms.distance_measures import resistance_distance
from scipy.stats import pearsonr


def unroll_inten(mean_int, ia, file_path, title):
    fontdict = {"fontsize": "xx-large", "fontweight": "semibold"}
    fontdict_lab = {"fontsize": 25, "fontweight": "bold"}

    fig, [ax, ax1] = plt.subplots(figsize=(16, 10), nrows=2, ncols=1, sharex=True)
    data = np.row_stack((utils.scale(mean_int, (0, 1)), utils.scale(ia, (0, 1))))
    im = ax.imshow(data, aspect="auto", origin="lower")
    cbar_label_dict = copy.deepcopy(fontdict_lab)
    cbar_label_dict.update(label="Mean Intensity")
    fig, ax, cbar = create_colorbar(
        fig,
        ax,
        mapp=im,
        array_of_values=data.reshape(-1),
        fontdict_cbar_tickslabel={"fontsize": fontdict_lab["fontsize"]},
        fontdict_cbar_label=cbar_label_dict,
    )
    # ax.set_yticks(np.arange(-.5, 1, 1), minor=True)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True", "Predict."], fontdict=fontdict_lab)
    # ax.set_xlabel('Nodes', fontdict=fontdict_lab)  # fontsize='x-large', fontweight='bold')
    ax.set_title(title, fontdict=fontdict_lab)

    err = data[1] - data[0]
    im2 = ax1.imshow(
        err[np.newaxis], aspect="auto", origin="lower", cmap=plt.cm.coolwarm
    )
    ax1.set_xlabel(
        "Nodes", fontdict=fontdict_lab
    )  # fontsize='x-large', fontweight='bold')
    ax1.set_yticks([1])
    cbar_label_dict.update(label="Err=Pred-True")
    # ax1.set_ylabel('Pred-True', fontdict=fontdict_lab)  # fontsize='x-large', fontweight='bold')
    fig, ax1, cbar1 = create_colorbar(
        fig,
        ax1,
        mapp=im2,
        array_of_values=err,
        fontdict_cbar_tickslabel={"fontsize": fontdict_lab["fontsize"]},
        fontdict_cbar_label=cbar_label_dict,
    )
    x_ticks = np.linspace(0, len(data[1]) - 1, 3, endpoint=True, dtype=int)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, fontdict=fontdict_lab)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def add_ground_node(graph, ground_node_x):
    ground_node = (
        "Gnd",
        {"coord": (ground_node_x + np.sign(ground_node_x) * 160, 250)},
    )
    connected_to_electrode = [
        (node, ground_node[0])
        for node, feat in graph.nodes(data=True)
        if feat["coord"][0] > ground_node_x
    ]
    graph.add_nodes_from([ground_node])
    graph.add_edges_from(connected_to_electrode, weight=1)


def effective_resistance(graph, ground_node_x):
    graph_grounded = copy.deepcopy(graph)
    add_ground_node(graph_grounded, ground_node_x)
    # graph_grounded = graph_grounded.subgraph(nx.shortest_path(graph_grounded, 'Ground'))
    # node_list = list(nx.shortest_path(graph_grounded, 'Ground').keys())
    # eff_res = [resistance_distance(G=graph_grounded, nodeA=node_name, nodeB=node_list[0], weight='weight', invert_weight=True)
    #           for node_name in node_list[1:]]# if node_name != 'Ground']
    eff_res = [
        resistance_distance(
            G=graph_grounded,
            nodeA=node_name,
            nodeB="Gnd",
            weight="weight",
            invert_weight=True,
        )
        for node_name in list(graph.nodes())
        if node_name != "Gnd"
    ]
    # visualize.plot_network(graph_grounded, node_color=[1/i for i in eff_res]+[0], show=True)
    return eff_res, 0  # node_list


def plot_error(graph, mean_int, pred, save_path):
    mean_int = utils.scale(mean_int, (0, 1))
    pred = utils.scale(pred, (0, 1))
    err = pred - mean_int
    corr, p_val = pearsonr(mean_int, pred)
    plt.hist(err)
    plt.ylabel("#")
    plt.xlabel("Err")
    plt.title(
        f"<Err=Pred-True>={err.mean():.3f},"
        f" std={np.std(err):.3f}\n<Abs(Err)>={np.abs(err).mean():.3f}"
    )
    plt.savefig(join(save_path, "hist_err_curr.png"))
    plt.close()

    plt.scatter(mean_int, pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Correlation {corr:.2f}, p-value {p_val:.2f}")
    plt.savefig(join(save_path, "scatter_intensity.png"))
    plt.close()

    visualize.plot_network(
        graph,
        node_color=err,
        title=f"Err = Pred - True. Corr={corr:.2f}",
        save_path=join(save_path, "error_map.png"),
        cmap=plt.cm.coolwarm,
    )


# def conductance_map(G):
#    def expanded_laplacian_block_matrix(G):
#        L = nx.laplacian_matrix(G=G, weight='weight')
#        L = L.todense()
#        ground_node = ('Gnd', {'coord': (520, 250)})
#        connected_to_electrode = np.array([1 if feat['coord'][0] > GROUND_NODE_X else 0 for node, feat in G.nodes(data=True)])
#        temp = np.hstack((L, connected_to_electrode[:, np.newaxis]))  # , axis=1)
#        L_exp = np.vstack((temp, np.append(connected_to_electrode, 0)))
#        return L_exp

#    intensities = list()
#    V = np.zeros(G.number_of_nodes() + 1)
#    L_exp = expanded_laplacian_block_matrix(G)
#    for tip_pos in range(G.number_of_nodes()):
#        V[tip_pos] = 1
#        I = np.dot(L_exp, V)
#        intensities.append(I[0, tip_pos])
#    return np.array(intensities)


# Sort folder by value
def sort_by_(string, list_dir):
    p_list = []
    for directory in list_dir:
        p_list.append(float(directory.rsplit(string, 1)[1]))
    ind = np.argsort(p_list)
    list_dir = np.asarray(list_dir)[ind]
    p_list = np.asarray(p_list)[ind]
    return p_list, list_dir


def markov_steady_state_proba(graph):
    """
    :param graph: Graph
    :return: Stationary vector for Markov Chain from graph G
    """
    # Adjacency matrix
    adj = nx.to_numpy_array(G=graph, weight="weight")
    # Create transition matrix row stochastic
    transition_matrix = adj / adj.sum(axis=1)[:, None]
    # We have to transpose (not if symmetric) so that Markov transitions correspond to
    # right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    eig_values, eig_vectors = np.linalg.eig(transition_matrix.T)
    eig_vectors_eig_val_1 = eig_vectors[:, np.isclose(eig_values, 1)]
    # Normalize and take real part. eigs finds complex eigenvalues
    # and eigenvectors, so you'll want the real part.
    stationary = (eig_vectors_eig_val_1 / eig_vectors_eig_val_1.sum()).real
    # Check stationarity
    # print(stationary == np.dot(stationary.T, transition_matrix).T)
    return stationary[:, 0]


def probability_of_state(graph):
    # For ergodic matrix only (connected components
    adj = nx.to_numpy_array(G=graph, weight="weight")
    degree_weighted = adj.sum(axis=1)
    # Check if symmetric
    # (adj == adj.T).all()
    w = degree_weighted / degree_weighted.sum()
    return w


def make_plots(
    graph, graph_info, plots_path, mean_int, inv_proba, inten_eff_res, ground_node_x
):
    corr_inv_proba = graph_info.corr_inv_proba
    n_segments = graph_info.slic_n_segments
    slic_sigma = graph_info.slic_sigma
    prune_proba = graph_info.prune_proba
    polar_avg_window = graph_info.polar_avg_window
    corr_eff_res = graph_info.corr_eff_res

    # Plots
    utils.ensure_dir(plots_path)
    weights_sequence = sorted(
        [d["weight"] for n1, n2, d in graph.edges(data=True)], reverse=True
    )
    plt.hist(weights_sequence)
    plt.savefig(join(plots_path, "weights_after_rescale.png"))
    # visualize.edge_weight_distribution(G)

    # P Inv
    utils.ensure_dir(join(plots_path, "Inv_Prob"))
    visualize.plot_network(
        graph,
        node_color=inv_proba,
        title=(
            f"Inv. Prob.. Corr={corr_inv_proba:.2f}\nSegm{n_segments:d},"
            f" sigma{slic_sigma:.1f}, p{prune_proba:.3f}, a{polar_avg_window:d}"
        ),
        save_path=join(plots_path, "Inv_Prob/eff_res_curr_map.png"),
        cmap=plt.cm.viridis,
    )
    unroll_inten(
        mean_int=mean_int,
        ia=inv_proba,
        file_path=join(plots_path, "Inv_Prob/pred_curr.png"),
        title=(
            f"Current map. Corr={corr_inv_proba:.2f}\nSegm{n_segments:d},"
            f" sigma{slic_sigma:.1f}, p{prune_proba:.3f}, a{polar_avg_window:d}"
        ),
    )
    plot_error(
        graph=graph,
        mean_int=mean_int,
        pred=inv_proba,
        save_path=join(plots_path, "Inv_Prob"),
    )

    # Eff Res
    utils.ensure_dir(join(plots_path, "EffRes"))
    g_grounded = copy.deepcopy(graph)
    add_ground_node(g_grounded, ground_node_x=ground_node_x)
    visualize.plot_network(
        graph,
        node_color=inten_eff_res,
        title=(
            f"Eff. Res.. Corr={corr_eff_res:.2f}\nSegm{n_segments:d},"
            f" sigma{slic_sigma:.1f}, p{prune_proba:.3f}, a{polar_avg_window:d}"
        ),
        save_path=join(plots_path, "EffRes/eff_res_curr_map.png"),
        cmap=plt.cm.viridis,
    )
    visualize.plot_network(
        g_grounded,
        node_color=inten_eff_res + [0],
        title=(
            f"Eff. Res.. Corr={corr_eff_res:.2f}\nSegm{n_segments:d},"
            f" sigma{slic_sigma:.1f}, p{prune_proba:.3f}, a{polar_avg_window:d}"
        ),
        save_path=join(plots_path, "EffRes/eff_res_curr_map_gnd.png"),
        cmap=plt.cm.viridis,
    )
    # visualize.plot_network(g_grounded, node_color=inten_eff_resi + [0], show=True)
    visualize.plot_network(
        graph,
        node_color=mean_int,
        title=(
            f"Ground Truth. \nSegm{n_segments:d}, sigma{slic_sigma:.1f},"
            f" p{prune_proba:.3f}, a{polar_avg_window:d}"
        ),
        save_path=join(plots_path, "EffRes/gr_truth.png"),
        cmap=plt.cm.viridis,
    )
    unroll_inten(
        mean_int=mean_int,
        ia=inten_eff_res,
        file_path=join(plots_path, "EffRes/pred_curr.png"),
        title=(
            f"Current map. Corr={corr_eff_res:.2f}\nSegm{n_segments},"
            f" sigma{slic_sigma:.1f}, p{prune_proba:.3f}, a{polar_avg_window}"
        ),
    )
    plot_error(
        graph=graph,
        mean_int=mean_int,
        pred=inten_eff_res,
        save_path=join(plots_path, "EffRes"),
    )


def compute_correlations(
    graph,
    base_graph_info,
    n_nodes_pre_pruning,
    self_loop,
    power_transf_w_min,
    plots_dir,
):
    if self_loop:
        lzw = list(
            dict([(node, (f["lzw"])) for node, f in graph.nodes(data=True)]).values()
        )
        lzw = utils.scale(lzw, (np.min(lzw), 1))
        self_loops = [(n, n, lzw[i]) for i, n in enumerate(graph.nodes) if lzw[i] > 0.7]
        graph.add_weighted_edges_from(ebunch_to_add=self_loops, weight="weight")
    weights = [w["weight"] for u, v, w in graph.edges(data=True)]

    rescale_weights(graph, (np.min(weights), 1))

    # if np.min(weights) < w_min:
    #    rescale_weights(G, scale=(w_min, 1), method='MinMax')
    # Weights Rescale
    if power_transf_w_min is not None:
        rescale_weights(graph, scale=(power_transf_w_min, 1), method="box-cox")

    mean_int = list(
        dict(
            [(node, (feat["mean_intensity"])) for node, feat in graph.nodes(data=True)]
        ).values()
    )
    mean_int = utils.scale(mean_int, (0, 1))

    # Invariant Probability
    inv_proba = markov_steady_state_proba(graph=graph)
    corr_inv_proba, p_val_inv_proba = pearsonr(mean_int, inv_proba)

    # Effective Resistance
    if "JR38_far" in base_graph_info.sample_name:
        ground_node_x = 480
    elif "JR38_close" in base_graph_info.sample_name:
        ground_node_x = 400
    else:
        ground_node_x = -1

    # visualize.plot_network(graph, show=True)
    eff_res, node_list = effective_resistance(graph, ground_node_x=ground_node_x)
    inten_eff_res = [1 / i for i in eff_res]
    # inten_eff_res = utils.scale(inten_eff_res, (0, 1))
    # visualize.plot_network(G, node_color=inten_eff_res, show=True)
    corr_eff_res, p_val_eff_res = pearsonr(mean_int, inten_eff_res)

    base_graph_info.corr_inv_proba = corr_inv_proba
    base_graph_info.p_val_inv_proba = p_val_inv_proba
    base_graph_info.corr_eff_res = corr_eff_res
    base_graph_info.p_val_eff_res = p_val_eff_res
    base_graph_info.size_percentage = (
        graph.number_of_nodes() / n_nodes_pre_pruning * 100
    )

    if plots_dir is not None:
        make_plots(
            graph,
            base_graph_info,
            plots_dir,
            mean_int,
            inv_proba,
            inten_eff_res,
            ground_node_x,
        )

    return base_graph_info


def create_dataset(path_, self_loop, power_transf_w_min, save_plots):
    list_elements = list()

    fold_list = next(os.walk(path_))[1]
    # fold_list = [fold_list[-1]]
    for f, sample_name in enumerate(fold_list):
        print("\n\n", sample_name)
        #    image = io.imread(list_img[f])
        #    pixels = np.shape(image)[0] * np.shape(image)[1]

        load_path_segm = join(path_, sample_name, "maxDist_diagonal_alpha_fixed")
        list_dir_segm = [f.name for f in os.scandir(load_path_segm) if f.is_dir()]
        segm_list, list_dir_segm = sort_by_(string="segments_", list_dir=list_dir_segm)
        # segm_list = segm_list[2:4]
        # list_dir_segm = list_dir_segm[2:4]
        for n_segments, dir_segm in zip(segm_list, list_dir_segm):
            n_segments = int(n_segments)
            print(f"Numb of segm {n_segments}")
            load_path_sigma = join(load_path_segm, dir_segm)
            list_dir_sigma = [f.name for f in os.scandir(load_path_sigma) if f.is_dir()]
            sigma_list, list_dir_sigma = sort_by_(
                string="sigma_", list_dir=list_dir_sigma
            )
            for slic_sigma, sigma_dir in zip(sigma_list, list_dir_sigma):
                load_path_a = join(load_path_sigma, sigma_dir)
                list_dir_a = [f.name for f in os.scandir(load_path_a) if f.is_dir()]
                a_list, list_dir_a = sort_by_(
                    string="polar_avg_window_", list_dir=list_dir_a
                )
                for polar_avg_window, a_dir in zip(a_list, list_dir_a):
                    polar_avg_window = int(polar_avg_window)
                    load_path_p = join(load_path_a, a_dir)
                    list_dir_p = [
                        f.name
                        for f in os.scandir(a_dir)
                        if f.is_dir() and "p_0" in f.name
                    ]
                    p_list, list_dir_p = sort_by_(string="p", list_dir=list_dir_p)
                    non_pruned_graph = nx.read_gml(join(load_path_p, "graph.gml"))
                    for prune_proba, p_dir in zip(p_list, list_dir_p):
                        print(sigma_dir, a_dir, p_dir)

                        p_path = join(load_path_p, p_dir)

                        graph = nx.read_gml(join(p_path, "graph.gml"))

                        graph_info = easydict.EasyDict()
                        graph_info.sample_name = sample_name
                        graph_info.n_segments = n_segments
                        graph_info.slic_sigma = slic_sigma
                        graph_info.edge_detection_algorithm = "sobel"  # Or Gaussian
                        graph_info.prune_proba = prune_proba
                        graph_info.n_nodes = graph.number_of_nodes()
                        graph_info.polar_avg_window = polar_avg_window

                        graph_info = compute_correlations(
                            graph,
                            graph_info,
                            non_pruned_graph.number_of_nodes(),
                            self_loop,
                            power_transf_w_min,
                            join(p_path, "plots") if save_plots else None,
                        )

                        with open(join(p_path, "correlation.json"), "w") as out_file:
                            json.dump(graph_info, out_file)

                        list_elements.append(graph_info)

    data = dict(enumerate(list_elements))
    dataframe = pd.DataFrame(data=data).T
    return dataframe


def process_graph(
    load_path: str,
    n_nodes_pre_pruning: int,
    self_loop: bool,
    power_transf_w_min: float,
    save_plots: bool,
):
    graph = nx.read_gml(join(load_path, "graph.gml"))
    args = PrunedGraphArgParser()
    args.load(join(load_path, "args.json"))
    join(os.path.dirname(load_path), "base_graph", "graph.gml")

    graph_info = easydict.EasyDict()
    graph_info.sample_name = args.sample_name
    graph_info.segmentation_algorithm = args.segmentation_alg.value
    graph_info.slic_n_segments = args.slic_n_segments
    graph_info.slic_sigma = args.slic_sigma
    graph_info.edge_detection_algorithm = args.edge_detection_alg.value
    graph_info.prune_proba = args.prune_proba
    graph_info.n_nodes = graph.number_of_nodes()
    graph_info.polar_avg_window = args.polar_avg_window

    graph_info = compute_correlations(
        graph,
        graph_info,
        n_nodes_pre_pruning,
        self_loop,
        power_transf_w_min,
        join(load_path, "plots") if save_plots else None,
    )

    with open(join(load_path, "correlation.json"), "w") as out_file:
        json.dump(graph_info, out_file)

    return graph_info


def process_pruned_graphs(
    data_dir: str,
    self_loop: bool,
    power_transf_w_min: float,
    save_plots: bool,
    incremental: bool,
):
    base_graph_dirs = glob.glob(join(data_dir, "**/base_graph"), recursive=True)
    for base_graph_dir in base_graph_dirs:
        # Get the number of base graph nodes before the graph was pruned
        n_nodes_pre_pruning = nx.read_gml(
            join(base_graph_dir, "graph.gml")
        ).number_of_nodes()
        # Get the pruned graph subfolder paths
        graph_dirs = [
            f.path
            for f in os.scandir(os.path.dirname(base_graph_dir))
            if f.is_dir() and "p_" in f.name
        ]
        # Process the pruned graphs
        for graph_dir in graph_dirs:
            # Skip if incremental and the correlation file already exists
            if incremental and os.path.exists(join(graph_dir, "correlation.json")):
                continue
            process_graph(
                graph_dir,
                n_nodes_pre_pruning,
                self_loop,
                power_transf_w_min,
                save_plots,
            )


def gather_graph_info(data_dir: str) -> pd.DataFrame:
    corr_json_paths = glob.glob(join(data_dir, "**/correlation.json"), recursive=True)
    return pd.DataFrame(
        data=(json.load(open(corr_json_file)) for corr_json_file in corr_json_paths)
    )


def main():
    parser = CreateDatasetArgParser()
    args = parser.parse_args()

    root_dir = abspath(join(".", os.pardir))
    load_path = join(root_dir, args.load_path)
    print("Data load path:", load_path)

    df_dir = join(root_dir, "data")
    df_save_path = join(df_dir, f"{args.df_name}.csv")
    utils.ensure_dir(df_dir)
    print("Dataframe path:", df_save_path)

    # dataframe = create_dataset(
    #     load_path, args.self_loop, args.power_transfer, args.make_plots
    # )

    if not args.compile_only:
        process_pruned_graphs(
            load_path,
            args.self_loop,
            args.power_transfer,
            args.save_plots,
            args.incremental,
        )
    dataframe = gather_graph_info(load_path)
    dataframe.to_csv(df_save_path)


if __name__ == "__main__":
    main()
