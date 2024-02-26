#!/bin/bash

save_fold="data/segmentation"

# Create the base graph
for seg in {200,250}; do
  for sigma in {15,20,40}; do
    echo "Num segm ${seg}, Sigma ${sigma}";

    python ./segmentation.py -segmAlg 'slic' -nSeg "$seg" -sigma "$sigma" \
        -edgeAlg 'gaussian_rotate_img' --save_fold "$save_fold" || \
        { echo "Segmentation script failed, quitting early"; exit 1; }
  done
done
