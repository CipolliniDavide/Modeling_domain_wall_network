# Sample Generation

This directory contains all the necessary scripts to generate artificial samples.
The `src` directory holds the code used to generate samples.

## Sample types

### Voronoi-like

In order to proxy how a configuration of domains and walls forms in a specific sample of the ferroelastic-ferroelectric 
BFO film, it's crucial to consider the growth and interaction of different domains and DWs during the formation process as the
sample seeks to find the most stable and energetically favorable configuration under given conditions.
To generate intricate shapes mirroring the structure of the BFO sample, every seed point is linked 
to either a vertically or horizontally scaled Chebyshev distance with a probability denoted by _p_. 
The resulting Voronoi diagram is considered _improper_ as the domains lack convexity. Furthermore, 
incorporating both vertically and horizontally scaled domains can result in their collision, 
potentially causing the formation of fractured domains.
Three parameters govern the tessellation: the
number of seed points _n_, the Chebyshev distance axial scale _β_, and
the probability, _p_, of a domain along either the vertical or horizontal
direction.

#### Generating samples

Samples can be generated with the `generate_voronoi_samples.py` script in this directory. A single sample can be generated as
follows:

```./generate_voronoi_samples.py --n_iterations $iter --n_samples $N --site_linear_density $d --p_horizontal $p --horizontal_beta $beta --sample_size $sample_size --save_binary path/to/sample_directory```


### Delaunay Triangulation

The Delaunay Triangulation type is an artificial sample type used to test the ability of our feature extraction system
to find lines and their angles.

#### Generating samples

Samples can be generated with the `generate_sample.py` script in this directory. A single sample can be generated as
follows:

```./generate_sample.py {sample_type} path/to/sample_directory```

This will create a sample in the target directory with the filenames `sample.png` and `sample.json` respectively, for
the image and description files.

If you wish to create multiple samples at once, supply the `--num_samples` parameter, followed by the number of samples,
as shown below:

```./generate_sample.py {sample_type} path/to/sample_directory --num_samples {number}```

This will create `number` samples in the target directory, with the sample files being named numerically, with `png`.

#### Parameters

The `sample_type` parameter shown in the examples above must match one of the available sample types (`voronoi`
or `delaunay`). Calling the script without any arguments will show the available options.

All other parameters are to customize sample generation are optional. The `-h` flag shows available parameters and their
description, when calling sample generation scripts with a correct sample type, i.e. `./generate_sample.py delaunay -h`.
