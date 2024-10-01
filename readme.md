This project contains all files necessary to reproduce the results in the article 
**_Modeling a domain wall network in BiFeO3 with stochastic geometry and entropy-based similarity measure_ Cipollini, Swierstra and Schomaker, Front. Mater., 23 January 2024
Sec. Semiconducting Materials and Devices
Volume 11 - 2024 | https://doi.org/10.3389/fmats.2024.1323153**

Raw data JR38_CoCr_6_70000.ibw included in this repository are the product of the experimental work of Jan Rieck and Prof. dr Beatriz Noheda already published in _Rieck et al. (2022). Ferroelastic Domain Walls in BiFeO3 as Memristive Networks. In Advanced Intelligent Systems. Wiley | https://doi.org/10.1002/aisy.202200292_.

Part of the code included in this repository was developed by Andele Swierstra as part of his Master Project at the University of Groningen.
The C++ code for junction detection is from _Junction detection in handwritten documents and its application to writer identification_, He, Wiering and Schomaker,
Pattern Recognition,
2015 | https://doi.org/10.1016/j.patcog.2015.05.022. Code available in Zenodo https://zenodo.org/records/10708418.

The manually annotated DW network dataset is in the folder [Dataset](Dataset)/[GroundTruth2](Dataset%2FGroundTruth2).
Data obtained through the optimization are included in the folder [data](data).

Data and figures utilized in the paper are gathered in the folder [data_and_figures_in_paper](data_and_figures_in_paper). Additionally, they can be accessed in [data](data) and [Figures_test](Figures_test), along with the results obtained by setting a larger number of iterations (n_iter=20) in the tessellation. For more details, refer to the comments in [run_geneticoptimization.sh](run_geneticoptimization.sh).

# How to reproduce results in the paper
1) Crop the samples to produce 25 crops:
`bash cropsample_fig2_figS1a.sh`
The bash also plots graphs on images and reproduces Fig2 A, B, C, and S1A, but it necessitates the crops to be annotated 
first (see point 2).

2) [graph_annotation](neuromorphic_materials/graph_annotation): **create the dataset** by:
   1) manually annotating **nodes**: 
   
      `python neuromorphic_materials/graph_annotation/annotate_nodes`
      
   2) manually annotating **edges**:
   
      `python neuromorphic_materials/graph_annotation/annotate_edges` 
   
      save them in [Dataset](Dataset)/[GroundTruth2](Dataset%2FGroundTruth2)/[nodes](Dataset%2FGroundTruth2%2Fnodes) and in [Dataset](Dataset)/[GroundTruth2](Dataset%2FGroundTruth2)/[edges](Dataset%2FGroundTruth2%2Fedges).
      
   3) **convert the edges to graph**:
      1) **Note**: **compile the junction detector**
         To make sure the junction detector is compiled for your processor (Linux/Windows/MacOs).
         Run from the terminal:
            
         `cd neuromorphic_materials/junction_graph_extraction`
      
         `python compile_beyond_ocr.py`
      
      2) Convert graphs in [edges](Dataset%2FGroundTruth2%2Fedges) to graphs in graphml format with script in folder [graph_annotation](neuromorphic_materials%2Fgraph_annotation):
         
         `python neuromorphic_materials/graph_annotation/annotation_to_graph.py 
          Dataset/GroundTruth2/edges 
          Dataset/GroundTruth2/graphml` 
      
         where former path is input path, i.e., the path to the folder used 
         to save the product of edge annotation;
         the latter path required is where graph are to be saved.


3) [graph_similarity](neuromorphic_materials/graph_similarity): run the genetic **optimization algorithm**:
    
   `bash run_geneticoptimization.sh`


   1) it runs the **optimization procedure** by `python ${root}/neuromorphic_materials/graph_similarity/scripts/gad_optimize_voronoi_generation.py`.
      Data are saved in [data](data)/[GroundTruth2_Grid](data%2FGroundTruth2_Grid).
      
   2) it runs `python ${root}/neuromorphic_materials/graph_similarity/scripts/make_plot_after_multiple_optimization_processes.py ` which does:
      1) the **analysis of optimization trajectories** 
         to compose Fig5 and S2.
      2) **Identifies gaussians to identify 
         the parameters** which are saved in `Figures_test/Results_GroundTruth2_Grid/optimization/.../GMM_solutions`.


4) **Generate Synthetic Samples** & **Degree and Spectral Analysis**.

   Once the parameters are identified, we can generate a synthetic dataset and compare it to the 
   empirical dataset by running:

   `bash generate_voronoi_samples.sh`

    The bash :
     - calls scripts in [sample_generation](neuromorphic_materials%2Fsample_generation): to **generate samples** in the form of images and networks.
     - creates Fig 4 (only middles steps).
     - It calls scripts in [graph_stat_analysis](neuromorphic_materials%2Fgraph_stat_analysis).\
       Which produces **the statistical analysis** of the spectral entropy and the degree distribution shown in Figure 6, 
       7 that are saved in [Figures_test](Figures_test)/[Results_GroundTruth2_Grid](Figures_test%2FResults_GroundTruth2_Grid)/[StatAnalysis](Figures_test%2FResults_GroundTruth2_Grid%2FStatAnalysis)
   
5) To produce grids with synthetic annotated samples run:
    `bash fig6c_figS1b.sh`

6) In [graph_stat_analysis](neuromorphic_materials%2Fgraph_stat_analysis) you can run:
   - [role_of_dataset_size.py](role_of_dataset_size.py) : to investigate on the effect of the dataset size. We can examine the fluctuations
   of the sample mean.
   In other words we look at the variance, 𝑉𝑎𝑟(𝑋‾𝑁), of the sample mean, 𝑋‾𝑁, for different sample
   sizes, N.
   - [FigS5_spectral_analysis_compare_random_grid_netw.py](FigS5_spectral_analysis_compare_random_grid_netw.py): to produce
         the figure S5 in the Supplementary material showcases the spectral entropy for three prototypical networks
         and Erdos-Renyi, a grid graph and a small-world network each one compared to the spectral entropy of the DW networks in 
         our dataset.

6) [scripts_for_figures](neuromorphic_materials%2Fscripts_for_figures) contains scripts to create grids of pngs and plot graphs on images. (See point 5)

7) `Fig1_spectral_entropy_illustration.py` produces the figures used to compose Fig1 in the paper.


# Set up your environment

Before installing the libraries in `requirements.txt` make sure that in the file `imageio==2.27.0` to avoid conflict with `scikit-image==0.21.0rc1`.

1. Install Python 3.11 from the Microsoft store or [this link](https://www.python.org/downloads/release/python-3113/). On Linux (assuming Ubuntu and friends), see [this askubuntu answer](https://askubuntu.com/a/682875). For other distros, ask your favourite search engine.
2. Install dependencies by running `python3.11 -m pip install -r requirements.txt`
3. Run the programs using `python3.11 <program_name>`, or `<program_name>` if `python3` points to Python 3.11 on your system.

**Note:** The programs might run on older versions of Python, but support is not guaranteed. 
About MacOS, the optimization does not work properly. The best choice is Linux and Python3.11 ;).

## Networkqit Patch
We replicate here info that can also be found in file ./graph_similarity/networkqit_patch.md.
A patch is necessary for networkqit to work with the dependencies listed in `requirements.txt`.

In `path/to/your/venv/lib/python3.11/site-packages/autograd/scipy/misc.py`, replace line 6 with:

`if not hasattr(osp_misc, 'logsumexp'):`.

# GPU samples generation
To use GPU for the sample generation you can change the following lines in `neuromorphic_materials/sample_generation/generate_voronoi_samples.py`:
1) line 12:
   `#from src.voronoi.generator_gpu import VoronoiGeneratorGPU`
   `from src.voronoi.generator import VoronoiGenerator`
3) line 33 changed to:
    #generator = VoronoiGeneratorGPU(
    generator = VoronoiGenerator(

In `neuromorphic_materials/sample_generation/voronoi_test.py` added:

`sys.path.insert(1, '{:s}/'.format(os.getcwd()))`
`#from src.voronoi.generator_gpu import VoronoiGeneratorGPU`
`from src.voronoi.generator import VoronoiGenerator`

Nevertheless, actual speed-up is not guaranteed.


# Extended Dataset for Thesis Chapter 4

This repository has been updated to contain the Dataset [GroundTruth3](Dataset/GroundTruth3), which contains the Grid dataset
[GroundTruth3](Dataset/GroundTruth3), but also the Rand dataset that includes random crops, and the mixed dataset of 40 samples
containing both grid and random crops.

I added the synthetic dataset [Voronoi128_d0.34_p0.16_beta2.94](Dataset/Voronoi128_d0.34_p0.16_beta2.94), 
the data produced during the genetic optimization [GroundTruth3_Grid_Rand](data/GroundTruth3_Grid_Rand),
and the figures obtained from the analysis of results [Results_GroundTruth3_GridRand](Figures_test/Results_GroundTruth3_GridRand).

# Zenodo
The synthetic data from the optimization procedure, the (extended) dataset, and the synthetic datasets are also uploaded in Zenodo:
https://zenodo.org/records/13866820
