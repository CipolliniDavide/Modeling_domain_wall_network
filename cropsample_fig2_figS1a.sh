#!/bin/bash

root=$(pwd)
echo ${root}
ds_fold="${root}/Dataset/GroundTruth2/"
figures_svp="Figures_test/fig2_S1a"
mkdir -p ${figures_svp}/


fig_format=pdf
############################ Crop raw data and make Figure 2a (and pngs for fig2b) ##################################
python neuromorphic_materials/scripts_for_figures/crop_sample_from_ibw.py -lp ${root}/JR38_CoCr_6_70000.ibw \
--save_path ${ds_fold} \
--wth_rand 0 \
--fig_format ${fig_format} \

echo Move figures 2a to ${root}/${figures_svp}
mv ${ds_fold}/*.p* ${root}/${figures_svp}


################################## Figure 2b ##################################################################
mkdir -p ${figures_svp}/img_for_fig2b/
cp ${ds_fold}/samples/GridLike/*_0.png $figures_svp/img_for_fig2b/
#
# Figure 2b
python neuromorphic_materials/scripts_for_figures/plot_png_grid.py \
-lp_img ${root}/${figures_svp}/img_for_fig2b/ \
--fig_name "fig2b" \
-svp ${root}/${figures_svp}/ \
-nc 2 -nr 2 \
--fig_format ${fig_format}

rm -r ${figures_svp}/img_for_fig2b/

############################################### Figure 2c ###############################################################
eps=1.4
python neuromorphic_materials/scripts_for_figures/visualize_graph_on_image.py \
-lp_g "$ds_fold/graphml/" \
-lp_img "$ds_fold/samples/GridLike/" \
-svp "$ds_fold/annotated_img2/" \
-eps ${eps} \
-n 25 \
--fig_format .png

cp ${ds_fold}annotated_img/0_0.png ${root}/${figures_svp}/fig2c.png
echo -e "\n\nFigure 2c is in folder ${root}/${figures_svp}\n\n"

########################################### Figure S1a ###################################################################
## Create figures for all (>50) manually annotated graphs : already done for fig 2c
#python neuromorphic_materials/scripts_for_figures/visualize_graph_on_image.py \
#-lp_g "$figures_svp/graphml/" \
#-lp_img "$figures_svp/samples/GridLike/" \
#-svp "$figures_svp/annotated_img/" \
#-eps ${eps} \
#-n 50 \
#--fig_format .png
#
## Create figure S1a depicting a grid 2x4 of manually annotated graphs
python neuromorphic_materials/scripts_for_figures/plot_png_grid.py \
-lp_img ${ds_fold}/annotated_img2/ \
--fig_name "figS1a" \
-svp ${root}/${figures_svp}/ \
-nc 4 -nr 2 \
--fig_format ${fig_format}
