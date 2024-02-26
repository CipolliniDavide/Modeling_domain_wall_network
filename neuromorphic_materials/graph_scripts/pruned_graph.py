#! /usr/bin/env python3

import os
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from os.path import abspath, join
from pathlib import Path

import helpers.superpixels

warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa

import helpers.graph as graph_utils
import helpers.utils as utils
import matplotlib
import networkx as nx
import numpy as np
from arg_parsers import PrunedGraphArgParser
from helpers.superpixels import SuperpixelGraph, get_segm_path_part


def prune_graph(p, base_graph, node_list, image_gray, segments, args, save_folder_base):
    args.prune_proba = p
    save_fold = join(save_folder_base, f"p_{args.prune_proba:.2f}")
    utils.ensure_dir(save_fold)
    # Save args on txt
    args.save(join(save_fold, "args.json"))

    # Make superpixel graph
    pruned_edge_list = graph_utils.pruning_of_edges(p=p, graph=base_graph)
    graph = graph_utils.convert_to_networkx(
        node_list=node_list, edge_list_with_attr=pruned_edge_list
    )
    graph = graph_utils.connected_components(graph)[0]

    # Community detection
    graph = graph_utils.community_detection(graph, save_fold, image_gray, segments)

    # Analysis of the largest connected component
    graph_utils.standard_analysis(
        image=image_gray,
        segments=segments,
        graph=graph,
        save_fold=save_fold,
        save_name=args.sample_name,
    )

    # Centrality
    graph_utils.centrality_plots(
        image=image_gray,
        segments=segments,
        graph=graph,
        save_fold=save_fold,
        save_name=args.sample_name,
    )

    # Save
    nx.write_gml(graph, join(save_fold, "graph.gml"), stringizer=str)
    del graph


def main():
    # define parser & its arguments
    parser = PrunedGraphArgParser()
    args = parser.parse_args()

    root_dir = abspath(join(".", os.pardir))
    image_path = Path(root_dir) / args.image_name
    save_folder_base = join(root_dir, args.save_fold)

    segm_path_part = get_segm_path_part(args)
    if segm_path_part is None:
        print("Segmentation algorithm is invalid, quitting.")
        exit(1)

    param_dir_path = (
        f"{args.sample_name}/{segm_path_part}/"
        f"edgeDetectAlg_{args.edge_detection_alg.value}/"
    )

    df_load_path = join(root_dir, args.df_load_path, param_dir_path)

    # Save path
    save_folder_base = join(
        save_folder_base,
        param_dir_path,
        f"maxDist_{args.max_distance}/alpha_{args.alpha_def:s}/polar_avg_window_{args.polar_avg_window}",
    )
    print(save_folder_base)
    utils.ensure_dir(save_folder_base)
    print("\n\n")
    # Load
    image_gray, image_rgb = utils.load_image(image_path)
    segments = np.load(join(df_load_path, "segments.npy"))

    # Create graph without any kind of pruning
    save_fold_base_gr = join(save_folder_base, "base_graph")
    utils.ensure_dir(save_fold_base_gr)
    sg = SuperpixelGraph(args.max_distance, args.alpha_def)
    sg.create_from_segmentation(
        load_dir=df_load_path,
        save_fold=save_fold_base_gr,
        avg_polar_window=args.polar_avg_window,
    )
    if args.save_polar_plots:
        sg.create_polar_plots(image_gray, Path(save_fold_base_gr))

    node_list = utils.pickle_load(join(save_fold_base_gr, "node_list"))
    edge_list = utils.pickle_load(join(save_fold_base_gr, "edge_list"))
    non_pruned_graph = graph_utils.convert_to_networkx(
        node_list=node_list, edge_list_with_attr=edge_list
    )
    nx.write_gml(non_pruned_graph, join(save_fold_base_gr, "graph.gml"), stringizer=str)

    # Some basic analysis and plots
    # Community detection
    non_pruned_graph = graph_utils.community_detection(
        non_pruned_graph, save_fold_base_gr, image_gray, segments
    )
    # Analysis of largest connected component
    graph_utils.standard_analysis(
        image=image_gray,
        segments=segments,
        graph=non_pruned_graph,
        save_fold=save_fold_base_gr,
        save_name=args.sample_name,
    )
    # Centrality
    graph_utils.centrality_plots(
        image=image_gray,
        segments=segments,
        graph=non_pruned_graph,
        save_fold=save_fold_base_gr,
        save_name=args.sample_name,
    )

    # From here on the pruning
    list_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Using matplotlib with multithreading requires the non-interactive "Agg" backend.
    matplotlib.use("Agg")
    with Pool(processes=cpu_count()) as pool:
        pool.map(
            partial(
                prune_graph,
                base_graph=non_pruned_graph,
                node_list=node_list,
                image_gray=image_gray,
                segments=segments,
                args=args,
                save_folder_base=save_folder_base,
            ),
            list_p,
        )


# table_imshow(load_path=save_folder_base, save_path=save_folder_base.rsplit('/',2)[0]+'/',
#             title='NumSegm={:d}, sigma={:.1f}'.format(args.n_segments, args.sigma))


if __name__ == "__main__":
    main()
