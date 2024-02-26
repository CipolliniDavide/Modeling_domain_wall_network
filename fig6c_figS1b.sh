#!/bin/bash

root=$(pwd)
echo "Root: $root"

figures_svp="Figures_test/fig6c_S1b"
mkdir -p ${figures_svp}/

fig_format=pdf

N=100
sample_size=128

d=0.32
p=0.20
beta=2.95
eps=1.4

load_fold="${root}/Dataset/Voronoi${sample_size}_d${d}_p${p}_beta${beta}"

# Plot graph over sample
python neuromorphic_materials/scripts_for_figures/visualize_graph_on_image.py \
-lp_g "$load_fold/graphml/" \
-lp_img "$load_fold/images/" \
-svp "$load_fold/annotated_img/" \
-eps ${eps} \
-n 50 \
--fig_format '.png'

################################################# Figure 6c #############################################################
python neuromorphic_materials/scripts_for_figures/plot_png_grid.py \
-lp_img "$load_fold/images/" \
--fig_name "6c" \
-svp "${root}/${figures_svp}/" \
-nc 6 -nr 2 \
--fig_format ${fig_format}


################################################# Figure S1b ##########################################################
# Place graphs on crops
#python neuromorphic_materials/scripts_for_figures/visualize_graph_on_image.py \
#-lp_g "$load_fold/graphml/" \
#-lp_img "$load_fold/images/" \
#-svp "$load_fold/annotated_img/" \
#-eps ${eps} \
#-n 50 \
#--fig_format '.png'

python neuromorphic_materials/scripts_for_figures/plot_png_grid.py \
-lp_img "$load_fold/annotated_img/" \
--fig_name "S1b" \
-svp "${root}/${figures_svp}/" \
-nc 4 -nr 2 \
--fig_format ${fig_format}

