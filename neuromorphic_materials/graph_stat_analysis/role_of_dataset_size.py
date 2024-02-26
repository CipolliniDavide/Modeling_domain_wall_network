"""
# Role of dataset size

To answer the reviewer's query about the role of the dataset size, we can examine the fluctuations
of the sample mean.
In other words we look at the variance, 𝑉𝑎𝑟(𝑋‾_𝑁), of the sample mean, 𝑋‾_𝑁, for different sample
sizes, N.

Our approach begins with the definition of an observable. Specifically, we opt for the
summation of the vector representing the spectral entropy, that is the functional that maps
each spectral entropy function to a numerical value—this serves as our chosen observable,
𝑋. The preference for a functional related to the spectral entropy stems from the
predominant reliance of our methods on this particular function. We remark that the sum of
the vector encoding the spectral entropy is akin to the integral of the spectral entropy.
We can now compute the sample mean, 𝑋‾_𝑁, of this observable multiple times (50), and look
at the fluctuations of the collected means around their mean value. This way we can quantify
how good the estimation of the population mean is when we estimate it as the sample mean
of N samples.

Finally, we plot it as a function of the sample size N (see figure). The computation is
undertaken over a synthetic dataset (of 500 samples) without loss of generality for the
statistical analysis as all along the manuscript it is evident how estimated deviations are
comparable between the ground truth dataset and the synthetic dataset.

"""

import os
import sys
sys.path.insert(1, '{:s}/'.format(os.getcwd()))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import levy
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from scipy.stats import gaussian_kde

from neuromorphic_materials.graph_scripts.helpers import visualize
from neuromorphic_materials.graph_scripts.helpers import utils, graph
from neuromorphic_materials.graph_scripts.helpers.visual_utils import set_ticks_label, create_order_of_magnitude_strings
from neuromorphic_materials.graph_scripts.helpers.kmeans_tsne import plot_clustered_points, plot_k_means_tsn_points

def variance_of_the_mean_around_the_mean_of_means(arrays_list, sample_sizes, num_samples = 50, type='mse'):
    '''
    Computes 50 means of variable X from samples of size N. {\bar{X}_N^(i)}_{i=1}^{i=50}.
    The variance of the sample mean around the mean of means for each sample size N is then computed,
    and the variance of the mean around the mean of means for each sample size is then computed and returned.
    Basically, we are computing the SEM for different samples of size N.

    :param arrays_list:
    :param sample_sizes: number of samples to compute each sample mean
    :param num_samples: number of mean fitnesses
    :return: the variance of the mean around the mean of means for each sample size is then computed and returned.
    '''

    result = {N: [] for N in sample_sizes}

    ind = np.arange(len(arrays_list))
    for N in sample_sizes:
        for s in range(num_samples):
            selected_arrays_ind = np.random.choice(ind, size=N, replace=True)
            # The random variables X_n are now the means (of the integrals of S)
            # X_n = (arrays_list[selected_arrays_ind]*np.log(tau_range)).sum(axis=1).mean()
            X_n = arrays_list[selected_arrays_ind].mean()
            result[N].append(X_n)
    var_arr = [np.var(result[N], ddof=1) / N for N in sample_sizes]
    # var_arr = [np.var(result[N], ddof=1) / N / np.mean(result[N]) for N in sample_sizes]

    return var_arr


def compute_mean_samples(arrays_list, sample_size, num_samples = 50, type='mse'):
    '''
    Computes 50 mean fitnesses from samples of size n
    :param arrays_list:
    :param sample_sizes: number of samples to compute each sample mean
    :param num_samples: number of mean fitnesses
    :return: list of mean
    '''

    result = {n: [] for n in sample_size}

    ind = np.arange(len(arrays_list))
    for n in sample_size:
        for _ in range(num_samples):
            selected_arrays_ind = np.random.choice(ind, size=n, replace=True)
            selected_arrays = arrays_list[selected_arrays_ind]
            mean_array = np.mean(selected_arrays, axis=0)

            pop_ent = np.mean(arrays_list, axis=0)
            if type == 'mse':
                mse = np.mean(np.power(pop_ent - mean_array, 2))
                result[n].append(mse)
            elif type == 'means':
                result[n].append(mean_array)
            else:
                raise
            # result[n].append(mean_array)
            # result[n].append(1/(np.sum(np.power(mean_array-spectral_ent_gt.mean(axis=0), 2))))

    return result

def variance_of_the_sample_mean(arrays_list, sample_size, desired_array):
    '''
    Compute the n fitness from a sample of size n. The n fitness can be used to compute how far is the sample mean fitness
    from the real mean fitness.

    :param arrays_list:
    :param sample_sizes:
    :return:
    '''

    # result = {n: [] for n in sample_sizes}
    result = list()
    ind = np.arange(len(arrays_list))
    for N in sample_size:
        # for n in range(N):
        selected_arrays_ind = np.random.choice(ind, size=N, replace=True)
        selected_arrays = arrays_list[selected_arrays_ind]
        # mean_array = np.mean(selected_arrays, axis=0)
        X_n_list = selected_arrays.sum(axis=1)
        X_N = X_n_list.mean()
        error = X_n_list - X_N
        # result[n].append(mean_array)
        mean_sqrd_err = np.power(error, 2).sum()/(N-1)
        variance_of_the_sample_mean = 1/N * mean_sqrd_err
        result.append(variance_of_the_sample_mean)
        # result[n].append(1 / (np.sum(np.power(mean_array - des_array, 2))))

    return result


if __name__ == "__main__":

    root = os.getcwd().rsplit('/', 2)[0] + '/'
    # root = './Users/dav/PycharmProjects/neuromorphic-materials/'
    # folder = root + 'Dataset/GroundTruth1/graphml'
    # save_folder = root + 'StatAnalysis/GroundTruth1/'
    # file_list_gt = sorted(glob('{:s}/*.graphml'.format(folder)))

    # folder = root + 'Dataset/VoronoiDataset256/graphml'
    # save_folder = root + 'StatAnalysis/GT_Vor256/'

    folder = root + 'Dataset/Var_Voronoi128_d0.32_p0.20_beta2.95/graphml'
    save_folder = folder.strip('graphml')
    utils.ensure_dir(save_folder)

    file_list_vor = sorted(glob('{:s}/*.graphml'.format(folder))) #[:200]

    # Load graphs
    graph_list_gt = list()
    graph_list_vor = list()
    crop_name = list()

    # for i, fgt in enumerate(file_list_gt):
    #     crop_name.append(fgt.rsplit('/')[-1].split('.')[0])
    #     graph_list_gt.append(nx.read_graphml(fgt))
    for i, fvor in enumerate(file_list_vor):
        graph_list_vor.append(nx.read_graphml(fvor))


    # Create lists
    spectral_ent_gt = list()
    spectral_ent_vor = list()
    log_min = -3  # Start value (10^-3)
    log_max = 4  # Stop value (10^4)
    num_points = 300  # Number of points

    # Generate the log-spaced array and compute the sum of the spectral entropy array
    tau_range = np.logspace(log_min, log_max, num=num_points)
    for gt in graph_list_gt:
        spectral_ent_gt.append(graph.entropy(L=nx.laplacian_matrix(gt).toarray(), beta_range=tau_range))
    for vor in graph_list_vor:
        spectral_ent_vor.append(graph.entropy(L=nx.laplacian_matrix(vor).toarray(), beta_range=tau_range))
    spectral_ent_gt = np.clip(spectral_ent_gt, 0, 1e16)
    spectral_ent_vor = np.clip(spectral_ent_vor, 0, 1e16)


    ###############################################################################################################
    # sample_sizes = [16, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    sample_sizes = np.arange(10, 100, 2)
    num_samples = 200
    np.random.seed(1)


    # We define the stocastic variable X
    X = spectral_ent_vor.sum(axis=1)
    var_arr = variance_of_the_mean_around_the_mean_of_means(arrays_list=X, sample_sizes=sample_sizes)

    fig, ax = plt.subplots()
    ax.plot(sample_sizes, var_arr)
    set_ticks_label(ax_type='x',
                    ax=ax,
                    data=sample_sizes,
                    ticks=[sample_sizes[0], 25, 35, 50, 60, 70, 80, 90, sample_sizes[-1]], num=5, valfmt="{x:.0f}",
                    ax_label='Sample size\nN')
    set_ticks_label(ax_type='y',
                    ax=ax,
                    data=var_arr,
                    num=6,
                    # ticks=[sample_sizes[0], 25, 35, 50, 60, 70, 80, 90, sample_sizes[-1]], num=5, valfmt="{x:.0f}",
                    ax_label='Var. of the sample mean\n' + r'$\mathbf{Var(\bar{X}_{N})}$')
    plt.tight_layout()
    plt.savefig(f'{save_folder}/variance_of_the_mean_of_means.png')
    plt.show()

    # var_of_sample_mean = variance_of_the_sample_mean(arrays_list=spectral_ent_vor,
    #                                                  sample_size=sample_sizes,
    #                                                  desired_array=spectral_ent_gt.mean(axis=0)
    #                                                  )
    # fig, ax = plt.subplots()
    # ax.plot(sample_sizes, var_of_sample_mean, '')
    # set_ticks_label(ax_type='x',
    #                 ax=ax,
    #                 data=sample_sizes,
    #                 ticks=[sample_sizes[0], 25, 35, 50, 60, 70, 80, 90, sample_sizes[-1]], num=5, valfmt="{x:.0f}",
    #                 ax_label='Sample size\nn')
    # plt.tight_layout()
    # plt.show()

