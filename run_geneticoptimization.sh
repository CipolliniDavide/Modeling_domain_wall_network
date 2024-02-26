#!/bin/bash

root=$(pwd)

########################################################################################################################
######################################### Optimization algorithm #######################################################
########################################################################################################################

# Smaller number of iterations (n_iterations=10) to reproduce the data in the paper: convergence is defined up to an arbitrary value.
# Cheaper in terms of computation but still stable in finding an optimal set of solutions.
# Larger number of iterations (n_iterations=20) sets full convergence at the cost of higher computational burden.
# A slightly different solution in terms of parameter values is found, however maintaining the same results in
# terms of spectral entropy and degree properties.

n_iterations=10
#n_iterations=20

# Repeat optimization n_start through n_end
n_start=1
n_end=15

num_generations=51
processes=8

# Merge nodes closer than eps in samples
eps=1.4

ds_name=GroundTruth2/graphml
data_svp=GroundTruth2_Grid

### The following lines run the optimization of the model's parameters
#for i in $(seq $n_start $n_end); do
#    echo "Optimization $i"
#
#    save_dir="${root}/data/${data_svp}/iter${n_iterations}_eps${eps}/voronoi_gads${i}/"
#
#    mkdir -p ${save_dir}
#
#    python ${root}/neuromorphic_materials/graph_similarity/scripts/gad_optimize_voronoi_generation.py \
#    ${root}/Dataset/${ds_name} \
#    ${root}/neuromorphic_materials/junction_graph_extraction/beyondOCR_junclets/my_junc \
#    ${save_dir} \
#    --merge_nodes_epsilon ${eps} \
#    --n_iterations ${n_iterations} \
#    --processes ${processes} \
#    --num_generations ${num_generations} \
#    > ${save_dir}/output.txt
#done


########################################################################################################################
############## Analysis of optimization trajectories and Gaussian mixture model to pick parameters #####################
########################################################################################################################

# Figure 5 and S2. Also saves solutions of GMM model to select optimal parameters.

load_path="${root}/data/${data_svp}/iter${n_iterations}_eps${eps}"
echo "Load data from ${load_path}"

figures_svp=Figures_test/Results_GroundTruth2_Grid

# Number of gaussians might change if new optimization process has been run. Look at the histogram
# and pick the desired number of gaussians heuristically.
num_gaussians=6
p_th=.6 # Might need to be adjusted if most of the trajectories converge to p<.5. In such a case choose something like p_th=.4 or .35.

python ${root}/neuromorphic_materials/graph_similarity/scripts/make_plot_after_multiple_optimization_processes.py \
-load ${load_path} \
-svp ${root}/${figures_svp}/optimization/iter${n_iterations}_eps${eps}/ \
-figform .pdf \
-n_gauss ${num_gaussians} \
-p_th ${p_th}

