
from . import utils

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.vq import kmeans, vq
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm


def plot_k_means_tsn_points(data, sample_names=None, k=3, save_fold=None, save_name='', perplexity=30, random_state=0,
                            visualize_3d=False, kmeans_before_tsne=False, figsize=(10, 8), point_size=30, show=False, custom_colors=None):
    # scaler = StandardScaler().fit(data)
    # scaled_data = scaler.transform(data)
    scaled_data = utils.scale(data, (-1, 1))
    # Check if perplexity is less than the number of samples
    if perplexity >= data.shape[0]:
        perplexity = data.shape[0] - 1

    z = TSNE(n_components=3 if visualize_3d else 2, perplexity=perplexity, random_state=random_state).fit_transform(
        scaled_data)

    if kmeans_before_tsne:
        centroids, _ = kmeans(scaled_data, k)
        idx, _ = vq(scaled_data, centroids)
    else:
        # scaler2 = StandardScaler().fit(z)
        # scaled_data2 = scaler2.transform(z)
        scaled_data2 = utils.scale(z, (-1, 1))
        centroids, _ = kmeans(scaled_data2, k)
        idx, _ = vq(scaled_data2, centroids)
        # idx = np.random.randint(0, k, size=data.shape[0])

    alpha = 0.7
    label_map = {l: i for i, l in enumerate(np.unique(idx))}
    node_colors = [label_map[target] for target in idx]

    if custom_colors is None:
        colormap = ListedColormap(plt.cm.jet(np.linspace(0, 1, k)))
    else:
        colormap = custom_colors[:k]

    fig = plt.figure(save_name + "-embedding", figsize=figsize)

    if visualize_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    scatter = ax.scatter(z[:, 0], z[:, 1], c=node_colors, cmap=colormap, alpha=alpha, s=point_size)

    if sample_names:
        for i, name in enumerate(sample_names):
            ax.annotate(name, (z[i, 0], z[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.axis("off")
    if visualize_3d:
        ax.set_title(save_name + f" - tSNE 3D + KMeans (Perplexity: {perplexity})", fontweight="bold", fontsize=15)
    else:
        ax.set_title(save_name + f" - tSNE 2D + KMeans (Perplexity: {perplexity})", fontweight="bold", fontsize=15)

    cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(k))
    cbar.set_label('Cluster', fontproperties=fm.FontProperties(weight='bold', size='xx-large'))
    cbar.ax.tick_params(labelsize='xx-large')

    plt.tight_layout()  # Adjust layout before saving

    if save_fold is not None:
        utils.ensure_dir(save_fold)
        if visualize_3d:
            plt.savefig(save_fold + save_name + "-tSNE_3d-kmeans{:02d}".format(k) + ".png")
        else:
            plt.savefig(save_fold + save_name + "-tSNE_2d-kmeans{:02d}".format(k) + ".png")

    if show:
        plt.show()
    else:
        plt.close()

    return idx  # Return the cluster labels for each sample


import numpy as np
import matplotlib.pyplot as plt
import os


def plot_clustered_points(locations, cluster_indices, save_fold=None, save_name=None, point_size=30, figsize=(10, 8),
                          show=False, colormap=None, sample_names=None):
    # Extract x and y coordinates from the location strings
    x_coords = [float(loc.split('_')[1]) for loc in locations]
    y_coords = [float(loc.split('_')[0]) for loc in locations]

    unique_clusters = np.unique(cluster_indices)
    num_clusters = len(unique_clusters)

    if colormap is None:
        colormap = plt.cm.get_cmap('tab20', num_clusters)

    # Create a scatter plot
    plt.figure(figsize=figsize)
    for i, cluster in enumerate(unique_clusters):
        mask = np.array(cluster_indices) == cluster
        plt.scatter(np.array(x_coords)[mask], np.array(y_coords)[mask], c=[colormap[i]], label=f'Cluster {cluster}',
                    s=point_size)
    if sample_names:
        for i, name in enumerate(sample_names):
            plt.annotate(name, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0, 5), ha='center')


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Clustered Points Plot')
    plt.legend()
    plt.grid()

    # Set both x and y axes on top
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('left')

    # Set the origin of the y-coordinates to the top left corner
    ax.invert_yaxis()
    # Adjust layout before saving or displaying
    plt.tight_layout()

    if save_fold is not None:
        os.makedirs(save_fold, exist_ok=True)
        plt.savefig(os.path.join(save_fold, save_name + ".png"))

    if show:
        plt.show()
    else:
        plt.close()
