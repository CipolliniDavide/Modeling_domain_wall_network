#! /usr/bin/env python3

import argparse
import copy
import os
from os.path import abspath, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helpers import utils
from helpers.visual import annotate_heatmap, create_colorbar


def table_mean_int_corr(
    correlation, sigma_list, p_list, label, title="", save_path="./"
):
    data = np.reshape(correlation, newshape=(len(sigma_list), len(p_list)))
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(data, cmap="plasma")
    fontdict = {"fontsize": "xx-large", "fontweight": "semibold"}
    fontdict_lab = {"fontsize": 23, "fontweight": "bold"}
    y_ticks = sigma_list
    x_ticks = p_list
    ax.set_xticks(np.arange(0.05, len(x_ticks) + 0.05))  # [-0.75,-0.25,0.25,0.75])
    ax.set_xticklabels(x_ticks, fontdict=fontdict_lab)
    ax.set_xlabel("p", fontdict=fontdict_lab)  # fontsize='x-large', fontweight='bold')
    ax.set_yticks(np.arange(0.05, len(y_ticks) + 0.05))  # [-0.75,-0.25,0.25,0.75])
    ax.set_yticklabels(y_ticks, fontdict=fontdict_lab)
    ax.set_ylabel(
        "sigma", fontdict=fontdict_lab
    )  # fontsize='x-large', fontweight='bold')
    # ax.set_title('Mean intensity {:s}'.format(label) + title, fontdict=fontdict_lab)
    ax.set_title("{:s}".format(title), fontdict=fontdict_lab)
    annotate_heatmap(
        im=im, valfmt="{x:.2f}", textcolors=("white", "black"), fontdict=fontdict
    )
    cbar_label_dict = copy.deepcopy(fontdict_lab)
    cbar_label_dict.update(label=label)
    create_colorbar(
        ax=ax,
        fig=fig,
        mapp=im,
        array_of_values=data,
        fontdict_cbar_label=cbar_label_dict,
    )
    plt.tight_layout()
    print("Saving table to:", save_path + "_table_intensity.png")
    plt.savefig(save_path + "_table_intensity.png")
    plt.close()


def distribution_corr_per_segm(
    data_list,
    suptitle,
    title_list,
    save_path,
    bins,
    key="corr_inv_proba",
    fontdict={"weight": "bold", "fontsize": "x-large"},
):
    fig, ax = plt.subplots(
        nrows=len(data_list[0]),
        ncols=len(data_list),
        sharex=True,
        sharey=True,
        figsize=(12, 10),
    )
    # Cheesy way to allow double indexing
    if len(data_list[0]) == 1:
        ax = np.array([ax])

    for j, good_data in enumerate(data_list):
        for i, d in enumerate(good_data):
            n_segments_list = np.unique(d["slic_n_segments"])
            for n_segments in n_segments_list:
                ax[i, 0].set_ylabel(
                    "{:s}\n#".format(d["sample_name"].iloc[0]), **fontdict
                )
                ax[i, j].tick_params(axis="y", labelsize=fontdict["fontsize"])
                ax[i, j].hist(
                    np.array(d.loc[d["slic_n_segments"] == n_segments][key]),
                    bins=bins,
                    alpha=0.5,
                    label="SuperPix {:d}".format(int(n_segments)),
                )
                # ax[i].hist(np.array(d['corr']))
        ax[-1, j].set_xlabel("Correlation", **fontdict)
        ax[-1, j].tick_params(axis="x", labelsize=fontdict["fontsize"])
        ax[0, j].set_title(title_list[j], fontdict)
        x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ax[i, j].set_xticklabels(x_ticks)
        ax[i, j].set_xticks(x_ticks)

    plt.legend(fontsize="xx-large")
    plt.suptitle(suptitle, **fontdict)
    plt.tight_layout()
    plt.savefig(save_path)


def distribution_corr(
    data_list,
    suptitle,
    title_list,
    save_path,
    bins,
    key="corr_inv_proba",
    fontdict={"weight": "bold", "fontsize": "x-large"},
):
    fig, ax = plt.subplots(
        nrows=len(data_list[0]),
        ncols=len(data_list),
        sharex=True,
        sharey=True,
        figsize=(14, 10),
    )
    # Cheesy way to allow double indexing
    if len(data_list[0]) == 1:
        ax = np.array([ax])

    for j, good_data in enumerate(data_list):
        for i, d in enumerate(good_data):
            # nsegm_list = np.unique(d["n_segments"])
            # for nsegm in nsegm_list:
            ax[i, 0].set_ylabel("{:s}\n#".format(d["sample_name"].iloc[0]), **fontdict)
            ax[i, j].tick_params(axis="y", labelsize=fontdict["fontsize"])
            ax[i, j].hist(np.array(d[key]), bins=bins, alpha=0.5)
            # ax[i].hist(np.array(d['corr']))
        ax[-1, j].set_xlabel("Correlation", **fontdict)
        ax[-1, j].tick_params(axis="x", labelsize=fontdict["fontsize"])
        ax[0, j].set_title(title_list[j], fontdict)
    x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax[i, j].set_xticklabels(x_ticks)
    ax[i, j].set_xticks(x_ticks)

    # plt.legend(fontsize="xx-large")
    plt.suptitle(suptitle, **fontdict)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_table_corr(
    df_, sample_name, load_path_segm, name_, title="", key="corr_eff_res"
):
    df = df_.loc[df_["sample_name"] == sample_name]
    n_segm_list = np.unique(df["slic_n_segments"])
    sigma_list = np.unique(
        df["slic_sigma"]
    )  # np.unique(df.loc[df['n_segm' == n_segm]]['sigma'])
    a_list = np.unique(df["polar_avg_window"])
    p_list = np.unique(df["prune_proba"])

    for n_segm in n_segm_list:
        for a in a_list:
            corr_list = [
                df.loc[
                    (df["slic_n_segments"] == n_segm)
                    & (df["slic_sigma"] == sigma)
                    & (df["polar_avg_window"] == a)
                    & (df["prune_proba"] == p)
                ][key]
                for sigma in sigma_list
                for p in p_list
            ]

            table_mean_int_corr(
                correlation=corr_list,
                sigma_list=sigma_list,
                p_list=p_list,
                label="correlation",
                title=title,  # , Regions{:d}'.format(int(numSegm)),
                save_path=join(
                    load_path_segm,
                    "segm{:d}_a{:d}_{:s}".format(int(n_segm), int(a), name_),
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataframe_path", default="data/dataset.csv")
    parser.add_argument("-s", "--sample_name", default="JR38_far")
    parser.add_argument("-n", "--save_dir", default="data/plots")
    args = parser.parse_args()

    root_dir = abspath(join(".", os.pardir))
    load_path = join(root_dir, args.dataframe_path)
    print("Dataframe path:", load_path)

    df = pd.read_csv(load_path, index_col=0)
    ################################

    # save_path_tables = save_path + '/hor_close/'
    # utils.ensure_dir(save_path_tables)
    # plot_table_corr(df, sample_name='hor_close', load_path_segm=save_path_tables, name_='eff_res',
    #                title='Corr. over effect. res.', key='corr_eff_res', )
    # plot_table_corr(df, sample_name='hor_close', load_path_segm=save_path_tables, name_='inv_prob',
    #                title='Corr. over Inv. Prob', key='corr', )

    #########################################
    save_dir = join(root_dir, args.save_dir)
    save_path_tables = join(save_dir, "tables")
    utils.ensure_dir(save_path_tables)
    plot_table_corr(
        df,
        sample_name=args.sample_name,
        load_path_segm=save_path_tables,
        name_="eff_res",
        title="Corr. over effect. res.",
        key="corr_eff_res",
    )
    plot_table_corr(
        df,
        sample_name=args.sample_name,
        load_path_segm=save_path_tables,
        name_="inv_prob",
        title="Corr. over Inv. Prob",
        key="corr_inv_proba",
    )

    # sample_list_temp = np.unique(df['sample'])
    # sample_list = [sample_list_temp[1], sample_list_temp[0], sample_list_temp[2]]
    sample_list = np.unique(df["sample_name"])
    list_df = [df.loc[df["sample_name"] == name] for name in sample_list]

    # 90% size
    title_list = ["Size_kept = 90%", "Size_kept = 90% & p_val<.1"]
    good_data_0 = [data.loc[data["size_percentage"] > 90] for data in list_df]

    good_data_1 = [
        data.loc[(data["p_val_inv_proba"] < 0.1) & (data["size_percentage"] > 90)]
        for data in list_df
    ]

    good_data_list = [good_data_0, good_data_1]
    bins = np.linspace(
        df.loc[df["size_percentage"] > 90]["corr_inv_proba"].min(),
        df.loc[df["size_percentage"] > 90]["corr_inv_proba"].max(),
    )

    distribution_corr_per_segm(
        data_list=good_data_list,
        bins=bins,
        title_list=title_list,
        key="corr_inv_proba",
        suptitle="Intensity proportional to invariant state Markov Chain",
        save_path=join(save_dir, "prob_state_perSegm.png"),
    )
    plt.close()
    distribution_corr(
        data_list=good_data_list,
        bins=bins,
        title_list=title_list,
        key="corr_inv_proba",
        suptitle="Intensity proportional to invariant state Markov Chain",
        save_path=join(save_dir, "prob_state.png"),
    )
    plt.close()

    ################################################################################################
    good_data_1 = [
        data.loc[(data["p_val_eff_res"] < 0.1) & (data["size_percentage"] > 90)]
        for data in list_df
    ]
    good_data_list = [good_data_0, good_data_1]
    distribution_corr_per_segm(
        data_list=good_data_list,
        bins=bins,
        title_list=title_list,
        key="corr_eff_res",
        suptitle="Eff Resistance",
        save_path=join(save_dir, "eff_res_perSegm.png"),
    )
    plt.close()

    distribution_corr(
        data_list=good_data_list,
        bins=bins,
        title_list=title_list,
        key="corr_eff_res",
        suptitle="Eff Resistance",
        save_path=join(save_dir, "eff_res.png"),
    )
    plt.close()


if __name__ == "__main__":
    main()
