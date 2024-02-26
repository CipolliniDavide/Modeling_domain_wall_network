#!/bin/bash

save_fold="data/segmentation"

# Create the base graph
for square_size in {16,32,64}; do
  echo "Square size ${square_size}";

  python ./segmentation.py -segmAlg 'squares' --square_size "$square_size" \
      -edgeAlg 'gaussian' --save_fold "$save_fold" || \
      { echo "Segmentation script failed, quitting early"; exit 1; }
done
