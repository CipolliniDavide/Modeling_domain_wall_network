import os
import sys

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two directories up
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm
import numpy as np
from neuromorphic_materials.graph_scripts.helpers.utils import ensure_dir


def plot_png_grid(png_files, load_img_path, n, nrows, ncols, output_path, figsize=(12, 8), enhance=False, low_percentile = 1,
    high_percentile = 40):
    '''

    By increasing the low_percentile, you can enhance the visibility of darker details, while
    decreasing the high_percentile can enhance the visibility of brighter details.
    Experiment with different percentile values to achieve the desired level of contrast improvement.

    :param png_files:
    :param n:
    :param nrows:
    :param ncols:
    :param output_path:
    :param figsize:
    :return:
    '''

    # Ensure n is not greater than the number of available '.png' files
    n = min(n, len(png_files))

    # Create a figure with nrows x ncols grid layout
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Iterate through the axes and load and plot the '.png' images
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            img_path = os.path.join(load_img_path, png_files[i])
            img = plt.imread(img_path)
            if enhance:
                # Enhance contrast by adjusting the intensity range
                min_val = np.percentile(img, low_percentile)  # 5th percentile
                max_val = np.percentile(img, high_percentile)  # 95th percentile

                # Apply the adjusted intensity range
                ax.imshow(img, cmap='gray', vmin=min_val, vmax=max_val)
            else:
                ax.imshow(img, cmap='gray')


            # ax.imshow(img, cmap=cm.gray)
            ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the plot in SVG format
    plt.savefig(output_path)

    # Show the plot (optional)
    if args.show_fig:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lp_img', '--load_img_path', default='../../Dataset/GroundTruth2/annoteted_img/',
                        type=str)
    parser.add_argument('-nc', '--ncols', default=4, type=int)
    parser.add_argument('-nr', '--nrows', default=2, type=int)
    parser.add_argument('-enh', '--enhance', default=False, type=bool)
    parser.add_argument('-svp', '--save_path', default='../../Figures/',
                        type=str)
    parser.add_argument('-name', '--fig_name', default='Vor',
                        type=str)
    parser.add_argument('-figform', '--fig_format', default='.pdf', type=str)
    parser.add_argument('-show', '--show_fig', default=False, type=bool)
    args = parser.parse_args()


    if (args.nrows == 2) & (args.ncols == 4):
        figsize = (16, 9)

    elif (args.nrows == 2) & (args.ncols == 6):
        figsize=(12, 4)
    else:
        figsize = (12, 12)


    ensure_dir(args.save_path)
    output_path = f'{args.save_path}/{args.fig_name}_grid_img{args.nrows}x{args.ncols}.{args.fig_format}'
    # Get a list of all '.png' files in the folder
    png_files = sorted([file for file in os.listdir(args.load_img_path) if file.endswith('.png')]) #[50:]
    print(f'Loaded images:\n{png_files[:args.nrows*args.ncols]}\n')
    plot_png_grid(png_files, load_img_path=args.load_img_path, n=args.nrows*args.ncols,
                  nrows=args.nrows, ncols=args.ncols,
                  output_path=output_path,
                  figsize=figsize, enhance=args.enhance)

    print(f'Figure saved to:\n\t{output_path}\n')

