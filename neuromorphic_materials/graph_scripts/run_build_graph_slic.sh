#!/bin/bash

save_fold="data/graphs"
df_load_path="data/segmentation"

# Create the base graph
for seg in {200,250}; do
  for sigma in {15,20,40}; do
    for a in {1,3,5}; do
      echo "Num segm $seg, Sigma $sigma, Polar avg window size $a"

      python ./pruned_graph.py -segmAlg 'slic' -edgeAlg 'sobel' -n_seg "$seg" \
        --slic_sigma "$sigma" --polar_avg_window "$a" --df_load_path "$df_load_path" \
        --max_d 'diagonal' --alpha_def 'fixed' --save_fold "$save_fold" \
        --sample_name JR38_far --image_name JR38_far.png || \
        { echo "Pruned graph script failed, quitting early"; exit 1; }
    done
  done
done
