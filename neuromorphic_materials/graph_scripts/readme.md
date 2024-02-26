# Graph Scripts

This folder contains the scripts that allow the angle extraction over crops of an image and produces polar histograms 
of the detected angles.

To run the current pipeline on a sample image, run the following scripts in order:

```bash
./segmentation.py
./pruned_graph.py
./create_dataset.py
./make_plots.py
```

All arguments are optional, with defaults provided. To see the arguments, use the script with the `-h` flag, or consult the code in the scripts or the argument parsers, which can be found in the `arg_parsers` directory.

The scripts shown above should be runnable in order using the default arguments. If this is not the case, or different arguments are provided to the first script, make sure that the arguments provided are matching for all scripts. The Gaussian angle extraction with image rotation is not yet fully implemented, so may not work yet with the pipeline.

## Testing angle extraction

To test angle extraction on one image, run

```./test_angle_extraction.py -i path/to/input_image.png```

This will use the default angle extraction algorithm (Gaussian with image rotation). Alternatively, place one or more images in the `data/angle_extraction_test` folder, and run

```./run_angle_extraction_tests.py```

to test angle extraction with all availalbe angle extraction methods (Gaussian, Sobel, Gaussian with image rotation).