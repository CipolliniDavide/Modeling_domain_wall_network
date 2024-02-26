#!/usr/bin/env python3
import itertools
import sys
from enum import Enum
from functools import partial
from multiprocessing import Pool, cpu_count
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from easydict import EasyDict
from matplotlib import pyplot as plt
from scipy.spatial import distance
from skimage.measure import regionprops

from . import angle_extraction, utils
from .angle_extraction import EdgeDetectionMethod
from .polar_plot import polar_plot


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_segm_path_part(args):
    segm_path_part = f"segmAlg_{args.segmentation_alg.value}"
    if args.segmentation_alg == SegmentationMethod.SLIC:
        segm_path_part += (
            f"_segments_{args.slic_n_segments}_sigma_{args.slic_sigma:.1f}"
        )
    elif args.segmentation_alg == SegmentationMethod.SQUARES:
        segm_path_part += f"_squareSize_{args.square_size}"
    else:
        return None

    return segm_path_part


class SegmentationMethod(Enum):
    SLIC = "slic"
    SQUARES = "squares"


class SuperpixelFeatures:
    def __init__(
        self, image, segments, bin_num=121, edge_alg=EdgeDetectionMethod.SOBEL
    ):
        self.image = image
        self.segments = segments
        self.bin_list = np.linspace(0, 360, num=bin_num, dtype=float)  # * np.pi / 180.0
        self.edge_alg = edge_alg

    def polar_features(self, image_cp, bin_list):
        features = None
        if self.edge_alg == EdgeDetectionMethod.SOBEL:
            angles, _ = angle_extraction.extract_pixel_angles_sobel(image_cp)
            features, _, _ = utils.empirical_pdf_and_cdf(angles, bins=bin_list)
        elif self.edge_alg == EdgeDetectionMethod.GAUSSIAN:
            angles, _ = angle_extraction.extract_pixel_angles_gaussian(image_cp, 5, 1)
            features, _, _ = utils.empirical_pdf_and_cdf(angles, bins=bin_list)
        elif self.edge_alg == EdgeDetectionMethod.GAUSSIAN_ROTATE_IMG:
            features = angle_extraction.extract_angles_gaussian_45_deg_eval(
                image_cp, 5, 1, 1.5
            )
        else:
            print("Invalid edge detection algorithm")
            exit(1)

        return features

    def super_pix(self, index):
        props = regionprops(self.segments, intensity_image=self.image)[index]
        dict_superpix = EasyDict()
        dict_superpix.label = props.label
        dict_superpix.coord_x = props.centroid[1]
        dict_superpix.coord_y = props.centroid[0]
        dict_superpix.mean_intensity = props.mean_intensity
        dict_superpix.intensity_max = props.intensity_max
        dict_superpix.equivalent_diameter = props.equivalent_diameter
        image_cp = np.empty_like(self.image)
        image_cp[:] = self.image[:]
        image_cp[self.segments != props.label] = 0

        # Lzw complexity
        dict_superpix.lzw = utils.lzw_complexity_img(image_cp)

        # Polar Features
        polar_feat = self.polar_features(image_cp, self.bin_list)

        return dict_superpix, polar_feat

    def create_multiprocess(self, save_fold, save_name):
        n_cpu = cpu_count()
        print("Number of cpu in use: ", n_cpu)
        indices = np.arange(len(np.unique(self.segments)))
        with Pool(processes=n_cpu) as pool:
            superpixels = pool.map(self.super_pix, indices)
        dict_pix = list()
        pol_feat = list()
        for d, p in superpixels:
            dict_pix.append(d)
            pol_feat.append(p)
        df = pd.DataFrame(dict_pix)
        utils.ensure_dir(save_fold)
        df.to_csv(join(save_fold, f"{save_name}.csv"))
        np.save(join(save_fold, "polar_feat"), pol_feat)
        np.save(join(save_fold, "bin_list"), self.bin_list)
        np.save(join(save_fold, "segments"), self.segments)


class SuperpixelGraph:
    def __init__(
        self, max_distance, alpha_def
    ):  # , image, segments, save_fold, max_distance='None', alpha_def='None'):
        self.segments = None
        self.dist_matrix = None
        self.adj = None
        self.bin_list = None
        self.polar_features = None
        self.lzw = None
        self.mean_diameter = None
        self.intensity_max = None
        self.mean_intensity = None
        self.coordinates = None
        self.label = None
        self.alpha_def = alpha_def
        self.max_distance = max_distance

    def find_neighbors(self):
        dist = distance.cdist(self.coordinates, self.coordinates, "euclidean")
        adj = None
        if self.max_distance == "none":
            print("None")
            adj = np.ones((self.n_superpixels, self.n_superpixels))
        elif self.max_distance == "diagonal":
            print("diagonal")
            adj = (dist < np.sqrt(2) * self.mean_diameter) * 1
        else:
            print("Invalid max distance")
            exit(1)
        np.fill_diagonal(adj, 0)
        # plt.imshow(adj);
        # plt.savefig('./adj');
        # plt.close()
        return adj, dist

    @staticmethod
    def clean_superpixel_polar_feats(
        polar_features: np.ndarray, conv_kernel: np.ndarray, reduce_spikes: bool
    ):
        # Rudimentary way to eliminate large spikes in the pdf, by repeatedly
        # replacing the largest value with an average of its neighbours
        if reduce_spikes:
            max_idx = len(polar_features) - 1
            for i in range(10):
                idx = np.argmax(polar_features)
                # print(n , idx, self.bin_list[idx])
                # Special case for last element to avoid out of range errors
                polar_features[idx] = (
                    polar_features[-2] + polar_features[0]
                    if idx == max_idx
                    else polar_features[idx - 1] + polar_features[idx + 1]
                ) / 2

        # We apply moving average if a window size is passed
        if conv_kernel is not None:
            # Using the 'wrap' mode to apply cyclic convolution
            # polar_features[n] = convolve(feat, conv_ker, mode='wrap')
            padding = conv_kernel.size // 2
            polar_features = np.convolve(
                np.concatenate(
                    (
                        polar_features[-padding:],
                        polar_features,
                        polar_features[:padding],
                    )
                ),
                conv_kernel,
                mode="valid",
            )

        return polar_features

    @staticmethod
    def clean_image_polar_feats(
        polar_features: np.ndarray, mov_avg_size: int, reduce_spikes: bool
    ):
        # If the mov_avg_size is one, there is no need to do any convolution
        conv_kernel = (
            np.ones(mov_avg_size) / mov_avg_size if mov_avg_size != 1 else None
        )

        for n, feat in enumerate(polar_features):
            # polar_features[n] = replace_with_mean_of_adjacent(feat)
            polar_features[n] = SuperpixelGraph.clean_superpixel_polar_feats(
                feat, conv_kernel, reduce_spikes
            )

        return polar_features

    def create_edge_connection_diagram(
        self, n_s, n_tar, image, save_fold: Path, zoom=False
    ):
        gamma, alpha, vec_s_to_tar = self.find_angles(n_s, n_tar)
        # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
        index_mask = self.create_rotational_mask_for_inequality(
            gamma=gamma, alpha=alpha, binning=self.bin_list
        )

        regions = regionprops(self.segments, intensity_image=image)
        props_tar = regions[n_tar]
        props_s = regions[n_s]
        image_cp = np.empty_like(image)
        image_cp[:] = image[:]
        image_cp[
            (self.segments != props_s.label) & (self.segments != props_tar.label)
        ] = 0
        dy = np.hstack((props_s.coords[:, 0], props_tar.coords[:, 0]))
        dx = np.hstack((props_s.coords[:, 1], props_tar.coords[:, 1]))

        title_dict = {"weight": "bold", "fontsize": "x-large"}
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(222)
        ax.imshow(image_cp, origin="lower")
        p_s_2_tar = (
            self.weight_formula(np.sum(self.polar_features[n_tar][index_mask]))
            * self.lzw[n_tar]
        )
        arrow_size = p_s_2_tar  # * 10 ** 3
        ax.set_title(
            (
                f"{props_tar.label - 1:d} --> {props_s.label - 1:d}\n"
                f"Weight={arrow_size:.2f}"
            ),
            title_dict,
        )
        ax.arrow(
            self.coordinates[n_tar][0],
            self.coordinates[n_tar][1],
            -vec_s_to_tar[0],
            -vec_s_to_tar[1],
            head_width=arrow_size,
            width=arrow_size,
            color="r",
        )
        # ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        #            arrowprops=dict(arrowstyle="->"))
        text_dict = {"weight": "bold", "color": "white", "fontsize": 15}
        if zoom:
            text_dict = {"weight": "bold", "color": "white", "fontsize": 30}
            ax.set_ylim(dy.min(), dy.max())
            ax.set_xlim(dx.min(), dx.max())
            # ax.scatter(self.coordinates[n_tar][0], self.coordinates[n_tar][1])
            # ax.scatter(self.coordinates[n_s][0], self.coordinates[n_s][1])
        ax.text(
            self.coordinates[n_tar][0],
            self.coordinates[n_tar][1],
            str(n_tar),
            **text_dict,
        )
        ax.text(
            self.coordinates[n_s][0], self.coordinates[n_s][1], str(n_s), **text_dict
        )
        polar_plot(
            self.polar_features[n_tar],
            mask=index_mask,
            # save_name='seg{:04d}_to_{:04d}'.format(n_tar, n_s), title='{:04d}->{:04d}'.format(n_tar, n_s),
            # save_fold=self.save_fold,
            title="Node {:d}".format(n_tar),
            geometry=224,
            fig=fig,
        )

        # Simply inverted n_s and n_tar
        ax = fig.add_subplot(221)
        ax.imshow(image_cp, origin="lower")
        gamma_inv, alpha_inv, vec_s_to_tar_inv = self.find_angles(n_s=n_tar, n_tar=n_s)
        # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
        index_mask_inv = self.create_rotational_mask_for_inequality(
            gamma=gamma_inv, alpha=alpha_inv, binning=self.bin_list
        )
        arrow_size_inv = (
            self.weight_formula(np.sum(self.polar_features[n_s][index_mask_inv]))
            * self.lzw[n_s]
        )
        ax.set_title(
            (
                f"{props_tar.label - 1:d} <-- {props_s.label - 1:d}\n"
                f"Weight={arrow_size_inv:.2f}"
            ),
            title_dict,
        )
        ax.arrow(
            self.coordinates[n_s][0],
            self.coordinates[n_s][1],
            -vec_s_to_tar_inv[0],
            -vec_s_to_tar_inv[1],
            head_width=arrow_size_inv,
            width=arrow_size,
            color="r",
        )
        text_dict = {"weight": "bold", "color": "white", "fontsize": 15}
        ax.text(
            self.coordinates[n_tar][0],
            self.coordinates[n_tar][1],
            str(n_tar),
            **text_dict,
        )
        ax.text(
            self.coordinates[n_s][0], self.coordinates[n_s][1], str(n_s), **text_dict
        )
        # if zoom:
        #    ax.set_ylim(dy.min(), dy.max())
        #    ax.set_xlim(dx.min(), dx.max())

        # ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        #            arrowprops=dict(arrowstyle="->"))
        polar_plot(
            self.polar_features[n_s],
            mask=index_mask_inv,
            save_filename=f"{n_s:03d}seg{n_s:04d}_to_{n_tar:04d}",
            # suptitle='{:02d}-{:02d}'.format(n_s, n_tar),
            title=f"Node {n_s:d}",
            save_dir=save_fold / "edge_connection_plots",
            geometry=223,
            fig=fig,
        )

    @staticmethod
    def create_polar_plot(idx, polar_feats, save_fold):
        polar_plot(data=polar_feats, save_filename=f"node_{idx}", save_dir=save_fold)

    def create_polar_plots(self, image, save_fold: Path, zoom: bool = False):
        # Make polar plots of individual nodes
        with Pool(processes=cpu_count()) as pool:
            print("Creating node polar plots")
            pool.starmap(
                self.create_polar_plot,
                zip(
                    itertools.count(),
                    self.polar_features,
                    itertools.repeat(save_fold / "node_polar_plots"),
                ),
            )

            # Plot the edge connection polar diagrams
            print("Creating edge connection polar diagrams")
            for n_s in range(len(self.coordinates)):
                target_list = np.argwhere(self.adj[n_s] == 1).reshape(-1)
                target_list = target_list[target_list > n_s]

                # TODO: Find better parallelisation method, this is still slow
                pool.starmap(
                    partial(
                        self.create_edge_connection_diagram,
                        image=image,
                        save_fold=save_fold,
                        zoom=zoom,
                    ),
                    ((n_s, n_tar) for n_tar in target_list),
                )

    def find_angles(self, n_s, n_tar):
        # Define a unitary vec from n_tar 2 source
        vec_s_to_tar = np.asarray(self.coordinates[n_tar]) - np.asarray(
            self.coordinates[n_s]
        )
        norm_vec_s_to_tar = vec_s_to_tar / np.sqrt(np.sum(vec_s_to_tar**2))
        # Find angle between unitary vec and reference vector
        gamma = np.arctan2(norm_vec_s_to_tar[1], norm_vec_s_to_tar[0])
        gamma = np.rad2deg(gamma) + 180
        # Find alpha to define the 2d solid angle
        if self.alpha_def == "fixed":
            alpha = np.abs(np.arctan(0.5 * self.mean_diameter / self.mean_diameter))
        elif self.alpha_def == "true":
            alpha = np.abs(
                np.arctan(0.5 * self.mean_diameter / self.dist_matrix[n_s, n_tar])
            )
        else:
            sys.exit(1)
        alpha = np.rad2deg(alpha)
        return gamma, alpha, vec_s_to_tar

    @staticmethod
    def weight_formula(value):
        w = -np.log(value)
        return 1 / w

    @staticmethod
    def create_rotational_mask_for_inequality(gamma, alpha, binning):
        binning = np.asarray(binning)
        bin_max = binning.max()
        bin_min = binning.min()
        # print(gamma-alpha, gamma+alpha)
        if (gamma + alpha) > bin_max:
            # print('Caso 1')
            # bin_bool = (binning < gamma + alpha - bin_max) | (binning >= bin_max - alpha)
            bin_bool = (binning < gamma + alpha - bin_max) | (binning >= gamma - alpha)
            # quello di prima
            # bin_bool= (binning < gamma + alpha - bin_max) | (binning >= gamma)
            # print(binning[bin_bool])
        elif (gamma - alpha) < bin_min:
            # print('Caso 2')
            bin_bool = (binning <= gamma + alpha) | (
                binning > bin_max - (alpha - gamma)
            )
            # print(binning[bin_bool])
        elif (gamma - alpha) > bin_min and (gamma + alpha) < bin_max:
            # print('Caso 3')
            bin_bool = (binning < (gamma + alpha)) & (binning >= (gamma - alpha))
        return bin_bool

    def create_edges(self):
        def sum_over_angles(n_s, n_tar):
            """
            :param n_s: TARGET
            :param n_tar: SOURCE
            :return:
            """
            gamma, alpha, vec_s_to_tar = self.find_angles(n_s, n_tar)
            # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
            index_mask = self.create_rotational_mask_for_inequality(
                gamma=gamma, alpha=alpha, binning=self.bin_list
            )
            p_s_2_tar = np.sum(self.polar_features[n_tar, index_mask])

            return p_s_2_tar

        edge_list = []
        edge_weight_list = []
        temperature_list = []
        # create vector pointing from point "a" to "b"
        for n_s, coord_source in enumerate(self.coordinates):
            target_list = np.argwhere(self.adj[n_s] == 1).reshape(-1)
            target_list = target_list[target_list > n_s]

            for n_tar in target_list:
                p_tar_2_s = sum_over_angles(n_s=n_s, n_tar=n_tar)  # * self.lzw[n_tar]
                p_s_2_tar = sum_over_angles(n_s=n_tar, n_tar=n_s)  # * self.lzw[n_s]
                # if n_s in [10, 7, 8, 50, 70, 13, 0, 26]:
                #    plot_method(n_s, n_tar, zoom=True)
                w_edge = self.weight_formula(p_tar_2_s) + self.weight_formula(p_s_2_tar)
                # w_edge = (weight_formula(p_tar_2_s) * self.lzw[n_tar] + weight_formula(p_s_2_tar) * self.lzw[n_s])
                # w_edge = self.mean_intensity[n_s]*(p_s_2_tar**.5) + self.mean_intensity[n_tar]*(p_tar_2_s**.5)
                edge_list = edge_list + [(coord_source, self.coordinates[n_tar])]
                edge_weight_list.append(w_edge)

        edge_weight_list = np.asarray(edge_weight_list)
        # plt.hist(self.edge_weight_list)
        # plt.savefig(self.save_fold+'before_pruning.png')
        # plt.close()
        return [
            (edge[0], edge[1], {"weight": w})
            for (edge, w) in zip(edge_list, edge_weight_list)
        ]

    def create_from_segmentation(self, load_dir, save_fold, avg_polar_window):
        # TODO: Reduce class variable initialisation in method
        df = pd.read_csv(join(load_dir, "df.csv"))
        self.label = list(df["label"])  # [props.label for props in self.regions]
        self.coordinates = [(x, y) for x, y in zip(df["coord_x"], df["coord_y"])]
        self.mean_intensity = np.asarray(df["mean_intensity"])
        # self.mean_intensity = np.asarray(self.mean_intensity)/np.sum(self.mean_intensity)
        self.intensity_max = np.asarray(df["intensity_max"])
        self.mean_diameter = np.mean(df["equivalent_diameter"])
        self.lzw = list(df["lzw"])
        self.segments = np.load(join(load_dir, "segments.npy"))
        polar_features = np.load(join(load_dir, "polar_feat.npy"))
        self.polar_features = self.clean_image_polar_feats(
            polar_features, avg_polar_window, False
        )

        self.bin_list = np.load(join(load_dir, "bin_list.npy"))[1:]
        self.adj, self.dist_matrix = self.find_neighbors()

        node_list = [
            (coord, {"x": feat, "mean_intensity": mean_i, "lzw": lzw, "coord": coord})
            for coord, feat, mean_i, lzw in zip(
                self.coordinates, self.polar_features, self.mean_intensity, self.lzw
            )
        ]
        edge_list_with_attr = self.create_edges()

        utils.pickle_save(
            filename=join(save_fold, "edge_list"), obj=edge_list_with_attr
        )
        utils.pickle_save(filename=join(save_fold, "node_list"), obj=node_list)

        # a = utils.pickle_load(filename=self.save_fold + 'edge_list')
