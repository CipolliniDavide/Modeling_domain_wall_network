#!/bin/bash

save_fold="data/graphs"
df_load_path="data/segmentation"

# Create the base graph
for square_size in {16,32,64}; do
  for a in {1,3,5}; do
    echo "Square size $square_size, Polar avg window size $a"

    python ./pruned_graph.py -segmAlg 'squares' --square_size "$square_size" \
      -edgeAlg 'gaussian' --polar_avg_window "$a" --df_load_path "$df_load_path" \
      --max_d 'diagonal' --alpha_def 'fixed' --save_fold "$save_fold" || \
      { echo "Pruned graph script failed, quitting early"; exit 1; }
  done
done
