# This folder contains: 
- [Fig1_spectral_entropy_illustration.py](../Fig1_spectral_entropy_illustration.py) : produces the figures that are used to compode Fig1 in the paper.
- [role_of_dataset_size.py](role_of_dataset_size.py) : To answer the reviewer's query about the role of the dataset size, we can examine the fluctuations
of the sample mean.
In other words we look at the variance, 𝑉𝑎𝑟(𝑋‾𝑁), of the sample mean, 𝑋‾𝑁, for different sample
sizes, N.
- [FigS5_spectral_analysis_compare_random_grid_netw.py](FigS5_spectral_analysis_compare_random_grid_netw.py): produces
the figure S5 in the Supplementary material showcasing the spectral entropy for three prototypical networks
and Erdos-Renyi, a grid-graph and a small-world network each one compared to the spectral entropy of the DW networks in 
our dataset.
- [degree_figures_compare.py](degree_figures_compare.py) to produce fig 7
- [spectral_analysis_compare.py](spectral_analysis_compare.py) to produce fig 6A,B
- [spectral_analysis.py](spectral_analysis.py) TODO: remove this script
- [degree_figures.py](degree_figures.py) TODO: remove this script
- [test_specific_heat.py](test_specific_heat.py) test script where the specific heat C=-dS/d(log tau) is computed for 
a grid graph. The computation is correct. 

Scripts [degree_figures_compare.py](degree_figures_compare.py) and [spectral_analysis_compare.py](spectral_analysis_compare.py)
are called by [generate_voronoi_samples.sh](..%2Fgenerate_voronoi_samples.sh). The latter is used to generate the 
dataset of synthetic samples and to run the statistical analysis on top and comparing to the dataset of empirical networks.