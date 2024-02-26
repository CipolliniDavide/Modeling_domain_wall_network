#!/bin/bash

root=$(pwd)
echo ${root}

N=100
#sample_size=256
sample_size=128
eps=1.4

####################################### PARAMETERS #####################################################################
############################ Grid dataset with 25 samples ##############################################################
# Short number of iterations to reproduce the data in the paper: convergence is defined up to an arbitrary value.
# Cheaper in terms of computation but still stable in finding an optimal set of solutions.

iter=10
d=0.32
p=0.20
beta=2.95

# Larger number of iterations sets full convergence. A different solution in terms of parameters values is found.
# This is more stable but more costly in terms of computational burden.

#n_iter=20
#d=0.38
#p=0.15
#beta=2.99

########################################################################################################################

save_fold="${root}/Dataset/Voronoi${sample_size}_d${d}_p${p}_beta${beta}"
echo "$save_fold"

############################ Generate the synthetic dataset ############################################################

#python neuromorphic_materials/sample_generation/generate_voronoi_samples.py \
#        --n_iterations $iter \
#        --n_samples $N \
#        --site_linear_density $d \
#        --p_horizontal $p \
#        --horizontal_beta $beta \
#        --sample_size $sample_size \
#        --save_binary "$save_fold/images"
#
## Extract the graph from the synthetic samples: Loop through numbers from 0 to N
#for i in $(seq 0 $(($N - 1))); do
#  echo "Extracting $i"
#  python neuromorphic_materials/junction_graph_extraction/scripts/extract_graph_voronoi.py \
#  "$save_fold/images/$i.png" \
#  "$save_fold/graphml/$i.graphml" \
#  neuromorphic_materials/junction_graph_extraction/beyondOCR_junclets/my_junc
#done

######################################### Fig 4 (only middles steps), (necessary for S1B, 6C) ##############################################################
## Fig 4
python neuromorphic_materials/scripts_for_figures/visualize_graph_on_image.py \
-lp_g "$save_fold/graphml/" \
-lp_img "$save_fold/images/" \
-svp "$save_fold/annotated_img/" \
-eps ${eps} \
-n 50 \
--fig_format '.png'

###########################################  RESULTS ###################################################################
################################# Plot analysis on graphs. Fig 7 and 6 #################################################
ds_name=GroundTruth2/graphml
ground_truth_ds=${root}/Dataset/${ds_name}

figures_svp=Figures_test/Results_GroundTruth2_Grid
svp_analysis=${root}/${figures_svp}/StatAnalysis/GT_Vor${sample_size}_d${d}_p${p}_beta${beta}

#show=True
figform=pdf

# Fig 7 A, B, C, D
python neuromorphic_materials/graph_stat_analysis/degree_figures_compare.py \
-lp_GT $ground_truth_ds \
-lp_VOR $save_fold/graphml/ \
-figform $figform \
-svp  $svp_analysis \
-eps ${eps}

##Fig 6 A, B
python neuromorphic_materials/graph_stat_analysis/spectral_analysis_compare.py \
-lp_GT $ground_truth_ds \
-lp_VOR $save_fold/graphml/ \
-figform $figform \
-svp ${svp_analysis} \
-eps ${eps}

