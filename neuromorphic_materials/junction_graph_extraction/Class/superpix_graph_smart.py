#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:58:05 2021

@author: hp
"""
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance
from skimage.measure import regionprops
from sklearn import preprocessing

from .polar_plot import polar_plot
from .utils import utils


class superpix_graph:
    def __init__(
        self, image, segments, save_fold, max_distance="None", alpha_def="None"
    ):
        self.image = image  # [..., 1]
        self.segments = segments
        self.n_superpixels = np.max(segments)
        self.save_fold = save_fold
        self.alpha_def = alpha_def
        utils.ensure_dir(save_fold)
        # parameter to erase all weights below a certain value.
        # p is the limit of the cdf to which is associated a value.
        # Edges smaller than that value are erased
        # self.p = p
        # Connect only pixels closer than d
        self.max_distance = max_distance
        # Create an undirected graph G
        # self.G = nx.Graph()
        self.create_graph()

    def find_neighbors(self, coords, mean_diameter):
        dist = distance.cdist(coords, coords, "euclidean")
        if self.max_distance == "None":
            print("None")
            Adj = np.ones((self.n_superpixels, self.n_superpixels))
        elif self.max_distance == "diagonal":
            print("diagonal")
            Adj = (dist < np.sqrt(2) * mean_diameter) * 1
        np.fill_diagonal(Adj, 0)
        # plt.imshow(Adj);
        # plt.savefig('./Adj');
        # plt.close()
        return Adj, dist

    def create_polar_features(self):
        def replace_with_mean_of_adjacent(data_list):
            for ind in range(len(data_list)):
                # ind = np.argmax(polar)
                # print(n , ind, self.bin_list[ind])
                if ind == len(data_list) - 1:
                    data_list[ind] = (data_list[-1] + data_list[0]) * 0.5
                elif ind == 0:
                    data_list[ind] = (data_list[-1] + data_list[0]) * 0.5
                else:
                    data_list[ind] = (data_list[ind - 1] + data_list[ind + 1]) * 0.5
            return data_list

        angles_features = np.asarray(
            [self.create_angle_features(props=props) for props in self.regions]
        )
        num_bins = 120
        polar_features = [
            utils.empirical_pdf_and_cdf(angles, bins=num_bins)[0]
            for angles in angles_features
        ]
        # polar_features_temp = polar_features
        # Rimuovo i max che probabilmente sono dovuti a sobel filter (almeno uno lo devo levare per forza se no non si vede nulla)
        # ker = [1, 0, 1]
        # polar_features_plot = [np.convolve(list(data), ker/np.sum(ker), mode='valid') for data in polar_features]
        for n, polar in enumerate(polar_features):
            # polar_features[n] = replace_with_mean_of_adjacent(polar)
            for i in range(10):
                ind = np.argmax(polar)
                # print(n , ind, self.bin_list[ind])
                if ind == len(polar) - 1:
                    polar_features[n][ind] = (polar[-1] + polar[0]) * 0.5
                elif ind == 0:
                    polar_features[n][ind] = (polar[-1] + polar[0]) * 0.5
                else:
                    polar_features[n][ind] = (polar[ind - 1] + polar[ind + 1]) * 0.5

        # Next line only to store bins (degree) used. It will be used in create_edges func.
        self.bin_list = np.linspace(
            0, 360, num=num_bins, dtype=float
        )  # * np.pi / 180.0

        # Plot lines
        seed = np.random.seed(1)
        indexs_to_plot = np.random.randint(0, len(polar_features), 10)
        # indexs_to_plot = [1, 50, 72]
        # indexs_to_plot = np.arange(len(polar))
        # We apply moving average
        mov_average = 3
        polar_features_plot = [
            np.convolve(list(data), np.ones(mov_average) / mov_average, mode="valid")
            for data in polar_features
        ]
        bin_list_plot = np.convolve(
            list(self.bin_list), np.ones(mov_average) / mov_average, mode="valid"
        )
        for i in indexs_to_plot:
            polar_plot(
                polar_features_plot[i],
                save_name="seg{:000d}_movavg{:d}".format(i, mov_average),
                save_fold=self.save_fold,
            )
            polar_plot(
                polar_features[i],
                save_name="seg{:000d}_withoutmax".format(i),
                save_fold=self.save_fold,
            )
            # polar_plot(polar_features_temp[i], save_name='seg{:000d}'.format(i), save_fold=self.save_fold)
        return polar_features  # polar_features_renormalized

    def create_angle_features(self, props):
        image_cp = np.empty_like(self.image)
        image_cp[:] = self.image[:]
        image_cp[self.segments != props.label] = 0

        # image_cp= image_cp.astype("float")
        # image_cp[self.segments != props.label] = np.nan

        # plt.imshow(image_cp, origin='lower')
        # plt.savefig('./' + str(props.label-1) + '.png')
        # Angles from sobel (degree) (-180,180]
        angles, _ = utils.angles_with_sobel_filter(image_cp)
        # We shift to (0, 360] and return
        return angles + 180

    def create_edges(self, mean_diameter):
        def plot_method(n_s, n_tar, zoom=False):
            gamma, alpha, vec_s_to_tar = find_angles(n_s=n_s, n_tar=n_tar)
            # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
            index_mask = create_rotational_mask_for_inequality(
                gamma=gamma, alpha=alpha, binning=self.bin_list
            )

            props_tar = self.regions[n_tar]
            props_s = self.regions[n_s]
            image_cp = np.empty_like(self.image)
            image_cp[:] = self.image[:]
            image_cp[
                (self.segments != props_s.label) & (self.segments != props_tar.label)
            ] = 0
            dy = np.hstack((props_s.coords[:, 0], props_tar.coords[:, 0]))
            dx = np.hstack((props_s.coords[:, 1], props_tar.coords[:, 1]))

            title_dict = {"weight": "bold", "fontsize": "x-large"}
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(222)
            ax.imshow(image_cp, origin="lower")
            p_s_2_tar = np.sum(self.polar_features[n_tar][index_mask])
            arrow_size = p_s_2_tar * 10**3
            ax.set_title(
                "{:d} --> {:d}\nWeight={:.2f} x10^-3".format(
                    props_tar.label - 1, props_s.label - 1, arrow_size
                ),
                title_dict,
            )
            ax.arrow(
                self.coordinates[n_tar][0],
                self.coordinates[n_tar][1],
                -vec_s_to_tar[0],
                -vec_s_to_tar[1],
                head_width=arrow_size * 8,
                width=arrow_size * 5,
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
                self.coordinates[n_s][0],
                self.coordinates[n_s][1],
                str(n_s),
                **text_dict,
            )
            polar_plot(
                self.polar_features[n_tar],
                mask=index_mask,
                # save_name='seg{:04d}_to_{:04d}'.format(n_tar, n_s), title='{:04d}->{:04d}'.format(n_tar, n_s),
                # save_fold=self.save_fold,
                title="Node {:d}".format(n_tar),
                geometry=(224),
                fig=fig,
            )

            # Simply inverted n_s and n_tar
            ax = fig.add_subplot(221)
            ax.imshow(image_cp, origin="lower")
            gamma_inv, alpha_inv, vec_s_to_tar_inv = find_angles(n_s=n_tar, n_tar=n_s)
            # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
            index_mask_inv = create_rotational_mask_for_inequality(
                gamma=gamma_inv, alpha=alpha_inv, binning=self.bin_list
            )
            arrow_size_inv = np.sum(self.polar_features[n_s][index_mask_inv]) * 10**3
            ax.set_title(
                "{:d} <-- {:d}\nWeight={:.2f} x10^-3".format(
                    props_tar.label - 1, props_s.label - 1, arrow_size_inv
                ),
                title_dict,
            )
            ax.arrow(
                self.coordinates[n_s][0],
                self.coordinates[n_s][1],
                -vec_s_to_tar_inv[0],
                -vec_s_to_tar_inv[1],
                head_width=arrow_size_inv * 8,
                width=arrow_size * 5,
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
                self.coordinates[n_s][0],
                self.coordinates[n_s][1],
                str(n_s),
                **text_dict,
            )
            # if zoom:
            #    ax.set_ylim(dy.min(), dy.max())
            #    ax.set_xlim(dx.min(), dx.max())

            # ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
            #            arrowprops=dict(arrowstyle="->"))
            polar_plot(
                self.polar_features[n_s],
                mask=index_mask_inv,
                save_name="{:03d}seg{:04d}_to_{:04d}".format(n_s, n_s, n_tar),
                # suptitle='{:02d}-{:02d}'.format(n_s, n_tar),
                title="Node {:d}".format(n_s),
                save_fold=self.save_fold,
                geometry=(223),
                fig=fig,
            )
            a = 0

        def find_angles(n_s, n_tar):
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
                # print('ciao')
                alpha = np.abs(np.arctan(0.5 * mean_diameter / mean_diameter))
            elif self.alpha_def == "None":
                alpha = np.abs(
                    np.arctan(0.5 * mean_diameter / self.dist_matrix[n_s, n_tar])
                )
            else:
                sys.exit(1)
            alpha = np.rad2deg(alpha)
            return gamma, alpha, vec_s_to_tar

        def create_rotational_mask_for_inequality(gamma, alpha, binning):
            binning = np.asarray(binning)
            bin_max = binning.max()
            bin_min = binning.min()
            # print(gamma-alpha, gamma+alpha)
            if (gamma + alpha) > bin_max:
                # print('Caso 1')
                # binbool = (binning < gamma + alpha - bin_max) | (binning >= bin_max - alpha)
                binbool = (binning < gamma + alpha - bin_max) | (
                    binning >= gamma - alpha
                )
                # quello di prima
                # binbool= (binning < gamma + alpha - bin_max) | (binning >= gamma)
                # print(binning[binbool])
            elif (gamma - alpha) < bin_min:
                # print('Caso 2')
                binbool = (binning <= gamma + alpha) | (
                    binning > bin_max - (alpha - gamma)
                )
                # print(binning[binbool])
            elif (gamma - alpha) > bin_min and (gamma + alpha) < bin_max:
                # print('Caso 3')
                binbool = (binning < (gamma + alpha)) & (binning >= (gamma - alpha))
            return binbool

        def sum_over_angles(n_s, n_tar):
            """
            :param n_s: TARGET
            :param n_tar: SOURCE
            :return:
            """
            gamma, alpha, vec_s_to_tar = find_angles(n_s=n_s, n_tar=n_tar)
            # Find those polar feature that are in between the 2d solid angle projected from n_source towards n_tar
            index_mask = create_rotational_mask_for_inequality(
                gamma=gamma, alpha=alpha, binning=self.bin_list
            )
            p_s_2_tar = np.sum(self.polar_features[n_tar][index_mask])

            return p_s_2_tar

        self.edge_list = []
        self.edge_weight_list = []
        temperature_list = []
        # create vector pointing from point "a" to "b"
        for n_s, coord_source in enumerate(self.coordinates):
            target_list = np.argwhere(self.Adj[n_s] == 1).reshape(-1)
            target_list = target_list[target_list > n_s]
            for n_tar in target_list:
                p_s_2_tar = sum_over_angles(n_s=n_s, n_tar=n_tar)
                p_tar_2_s = sum_over_angles(n_s=n_tar, n_tar=n_s)
                if n_s in [10, 7, 8, 50, 70, 13, 0, 26]:
                    plot_method(n_s, n_tar, zoom=True)
                w_edge = np.log(p_s_2_tar + p_tar_2_s)
                # w_edge = self.mean_intensity[n_s]*(p_s_2_tar**.5) + self.mean_intensity[n_tar]*(p_tar_2_s**.5)
                self.edge_list = self.edge_list + [
                    (coord_source, self.coordinates[n_tar])
                ]
                self.edge_weight_list.append(w_edge)
        # Scale weights between [0,1]
        self.edge_weight_list = utils.scale(self.edge_weight_list, (0, 1))
        print("# weights ", len(self.edge_weight_list))
        # Make weight matrix sparse
        plt.hist(self.edge_weight_list)
        plt.savefig(self.save_fold + "prima_del_pruning.png")
        plt.close()
        # Normalize weights
        # self.edge_weight_list = self.edge_weight_list / np.sum(self.edge_weight_list)
        # plt.hist(self.edge_weight_list)
        # plt.savefig(self.save_fold + 'dopo_pruning+norm.png')
        # plt.close()
        print(
            "Weights sum",
            self.edge_weight_list.sum(),
            ". Len:",
            len(self.edge_weight_list),
        )
        # self.edge_weight_list= np.exp(self.edge_weight_list)/sum(np.exp(self.edge_weight_list))
        return [
            (edge[0], edge[1], {"weight": w})
            for (edge, w) in zip(self.edge_list, self.edge_weight_list)
        ]

    def pruning_of_edges(self, p, save_fold="./"):
        def eps_edges(p=0.25):
            pdf, cdf, bins = utils.empirical_pdf_and_cdf(
                self.edge_weight_list, bins=100
            )
            plt.close()
            try:
                return bins[cdf < p][-1]
            except:
                return 0

        # Find weights to keep: weights > epsilon
        epsilon = eps_edges(p=p)
        # plot cdf
        pdf, cdf, bins = utils.empirical_pdf_and_cdf(self.edge_weight_list, bins=100)
        plt.close()
        plt.plot(bins, cdf)
        plt.axvline(epsilon, c="red")
        plt.xlabel("Edge weight")
        plt.ylabel("CDF")
        plt.savefig(save_fold + "CDF_edges.png")
        plt.close()
        #
        edges_ind_to_keep = [
            ind for ind, w in enumerate(self.edge_weight_list) if w > epsilon
        ]
        edge_weight_list = np.array(
            [self.edge_weight_list[i] for i in edges_ind_to_keep]
        )  # np.array(self.edge_weight_list)[edges_ind_to_keep]
        edge_list = [self.edge_list[i] for i in edges_ind_to_keep]
        # print('Weights smaller than epsilon={} are erased. \n\tMin weight: {}. Max weight: {}'.format(epsilon, np.min(
        #    edge_weight_list), edge_weight_list.max()))
        # print('\tCheck: {}'.format(edge_weight_list.min() >= epsilon))
        # plt.hist(edge_weight_list)
        # plt.savefig(save_fold + 'dopo_il_pruning.png')
        # plt.close()
        # Normalize weights
        # self.edge_weight_list = self.edge_weight_list / np.sum(self.edge_weight_list)
        # plt.hist(self.edge_weight_list)
        # plt.savefig(self.save_fold + 'dopo_pruning+norm.png')
        # plt.close()
        # print('Weights sum', edge_weight_list.sum(), '. Len:', len(edge_weight_list))
        # self.edge_weight_list= np.exp(self.edge_weight_list)/sum(np.exp(self.edge_weight_list))
        return [
            (edge[0], edge[1], {"weight": w})
            for (edge, w) in zip(edge_list, edge_weight_list)
        ]

    def create_graph(self):
        # image= image[..., 1]
        self.regions = regionprops(self.segments, intensity_image=self.image)
        equivalent_diameter = [props.equivalent_diameter for props in self.regions]
        self.mean_diameter = np.mean(equivalent_diameter)
        # ax_minor_lenght = [props.equivalent_diameter for props in regions]

        self.lzw = []
        for props in self.regions:
            image_cp = np.empty_like(self.image)
            image_cp[:] = self.image[:]
            image_cp[self.segments != props.label] = 0
            # print(props.label)
            # plt.imshow(image_cp)
            # plt.show()
            self.lzw.append(utils.LZW_complexity_img(image_cp))
        del image_cp
        self.label = [props.label for props in self.regions]
        self.coordinates = [
            (props.centroid[1], props.centroid[0]) for props in self.regions
        ]
        self.mean_intensity = [props.mean_intensity for props in self.regions]
        self.mean_intensity = np.asarray(self.mean_intensity) / np.sum(
            self.mean_intensity
        )

        self.Adj, self.dist_matrix = self.find_neighbors(
            self.coordinates, self.mean_diameter
        )
        self.polar_features = self.create_polar_features()
        self.node_list = [
            (coord, {"x": feat, "mean_intensity": mean_i, "lzw": lzw, "coord": coord})
            for coord, feat, mean_i, lzw in zip(
                self.coordinates, self.polar_features, self.mean_intensity, self.lzw
            )
        ]
        self.edge_list_with_attr = self.create_edges(self.mean_diameter)

        """# Nodes
        for props in self.regions:
            cx, cy = props.centroid  # centroid coordinates
            v = props.label  # value of label
            print(v)
            mean_a = props.mean_intensity
            image_cp= np.empty_like(image)
            image_cp[:]= image[:] 
            image_cp[segments != props.label]= 0
            #print(props.label)
            #plt.imshow(image_cp)
            #plt.show()
            lzw= utils.LZW_complexity_img(image_cp)
            
            G.add_node(v, cx= cx, cy= cy, mean_intensity= mean_a, lzw_complexity=lzw)
        #for node in self.G.nodes(data=True): print(node)
        #print(nx.get_node_attributes(G, 'cx')[1])

        #coords= np.asarray(coords)
        #dist= distance.cdist(coords, coords, 'euclidean')
        #ang_dist= np.arctan(dist[:,1]/dist[:,0])
        # Edges
        #color=nx.get_node_attributes(G,'cx')
        for node_source in list(G.nodes(data=True)):
            for node_target in list(G.nodes(data=True)):
                if node_source[0] is not node_target[0]:
                    distance_2= ( ( node_source[1]['cx'] - node_target[1]['cx'])**2 + 
                                (node_source[1]['cy'] - node_target[1]['cy'])**2 )
                    #weight= np.exp( - distance_2)
                    weight= node_source[1]['mean_intensity']*node_target[1]['mean_intensity']/distance_2
                    #print(node_source[0], node_target[0])
                    #print(weight)
                    #print(node_source[1]['cx'], node_target[1]['cx'])
                    #print(node_source[1]['cx'] - node_target[1]['cx'])
                    G.add_edge(node_source[0], node_target[0], weight= weight)
                #else: print(node_source[0], node_target[0])
        """
        # return G


def rag(image, labels, connectivity=2, weighted="True"):
    import numpy as np
    from skimage.future.graph import RAG

    # initialize the RAG
    graph = RAG(labels, connectivity=connectivity)
    # lets say we want for each node on the graph a label, a pixel count and a total color
    graph = add_attr_(image, labels, graph)
    # for n in graph:
    # graph.node[n].update({'labels': [n],'pixel count': 0,
    #                     'total color': np.array([0, 0, 0],
    #                     dtype=np.double)})

    # give them values
    # for index in np.ndindex(labels.shape):
    #    current = labels[index]
    #    graph.node[current]['pixel count'] += 1
    #    graph.node[current]['total color'] += image[index]

    if weighted:
        # calculate your own weights here
        for x, y, d in graph.edges(data=True):
            distance_2 = (graph.nodes[x]["cx"] - graph.nodes[y]["cx"]) ** 2 + (
                graph.nodes[x]["cy"] - graph.nodes[y]["cy"]
            ) ** 2
            # weight= np.exp( - distance_2)
            weight = (
                graph.nodes[x]["mean_intensity"]
                * graph.nodes[y]["mean_intensity"]
                / distance_2
            )
            d["weight"] = weight

    return graph


def add_attr_(image, segments, G):
    from skimage.color import rgb2gray

    image = utils.scale(rgb2gray(image), (0, 255))
    # image = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    regions = regionprops(segments, intensity_image=image)
    for props in regions:
        # cx, cy = props.centroid  # centroid coordinates
        v = props.label  # value of label
        print(v)
        mean_a = props.mean_intensity
        image_cp = np.empty_like(image)
        image_cp[:] = image[:]
        image_cp[segments != props.label] = 0
        # print(props.label)
        # plt.imshow(image_cp)
        # plt.savefig('./lzw_'+str(v))
        # plt.close()
        # plt.show()
        lzw = utils.LZW_complexity_img(image_cp)
        # print('lzw', v)
        # print(lzw)
        dict_attr = {"mean_intensity": mean_a, "lzw_complexity": lzw}
        # print(G.nodes[v])
        G.add_node(v, dict_attr)
        # print(G.nodes[v])


def remove_attribute(G, attr):
    for n in G.nodes:
        G.nodes[n].pop(attr, None)


"""
import torch
import torch.nn as nn
from torchvision import models
model = models.vgg.vgg16(pretrained=True)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
"""


def add_attr_Cnn(image, segments, G, save_path):
    from skimage.color import gray2rgb, rgb2gray

    image = utils.scale(rgb2gray(image), (0, 255))
    # image = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
    cnn_feat = []
    regions = regionprops(segments, intensity_image=image)
    for props in regions:
        cx, cy = props.centroid  # centroid coordinates
        v = props.label  # value of label
        print(v)
        mean_a = props.mean_intensity
        image_cp = np.empty_like(image)
        image_cp[:] = image[:]
        image_cp[segments != props.label] = 0
        # print(props.label)
        # plt.imshow(image_cp)
        # plt.savefig('./lzw_'+str(v))
        # plt.close()
        # plt.show()
        lzw = utils.LZW_complexity_img(image_cp)
        # print('lzw', v)
        # print(lzw)

        model.fc = Identity()
        # x = torch.randn(1, 3, 64, 64)
        image_cp_rgb = gray2rgb(image_cp)
        x = torch.nn.functional.interpolate(
            torch.tensor(image_cp_rgb, dtype=torch.float32).reshape(
                (1, 3) + image_cp_rgb.shape[:2]
            ),
            size=(64, 64),
        )  # , mode='bilinear', align_corners=False)
        # output = model.features(x)
        output = model.features[:5](x).flatten().detach().numpy()
        cnn_feat.append(output)
        print(output.shape)
        # for i in range(100):
        #    plt.imshow(output.detach().numpy()[0, i])
        #    plt.savefig('./features{:03d}.png'.format(i))
        dict_attr = {
            "cx": cx,
            "cy": cy,
            "mean_intensity": mean_a,
            "lzw_complexity": lzw,
        }
        # print(G.nodes[v])
        G.add_node(v, dict_attr)
    np.save(save_path + "cnn_features.npy", cnn_feat)

    # print(G.nodes[v])
