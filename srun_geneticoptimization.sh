#!/bin/bash

#SBATCH --mail-type=BEGIN
#SBATCH --job-name='gad_optim'
#SBATCH --output=gad_optim-%j.log
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-12:00
#SBATCH --mem=20G
#SBATCH --array=16-30

source $(pwd)/virtual_env/Modeling_domain_wall_network/bin/activate

ds_name=GroundTruth2/graphml
data_svp=GroundTruth2_Grid

eps=1.4

# Smaller number of iterations (n_iterations=10) to reproduce the data in the paper: convergence is defined up to an arbitrary value.
# Cheaper in terms of computation but still stable in finding an optimal set of solutions.
# Larger number of iterations (n_iterations=20) sets full convergence at the cost of higher computational burden.
# A slightly different solution in terms of parameter values is found, however maintaining the same results in
# terms of spectral entropy and degree properties.

n_iterations=10
#n_iterations=20

num_generations=51
processes=10

cd ./Modeling_domain_wall_network/
root=$(pwd)
echo "Root ${root}"

i=${SLURM_ARRAY_TASK_ID}

echo "Optimization $i"
save_dir="${root}/data/${data_svp}/iter${n_iterations}_eps${eps}/voronoi_gads${i}/"
echo "Save dir: ${save_dir}"
mkdir -p ${save_dir}

python ./neuromorphic_materials/graph_similarity/scripts/gad_optimize_voronoi_generation.py \
${root}/Dataset/${ds_name} \
${root}/neuromorphic_materials/junction_graph_extraction/beyondOCR_junclets/my_junc \
${save_dir} \
--merge_nodes_epsilon ${eps} \
--n_iterations ${n_iterations} \
--processes ${processes} \
--num_generations ${num_generations} \
> ${save_dir}output.txt

