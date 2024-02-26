import sys
import os
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate two directories up
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
import copy
import matplotlib.image
from igor2 import binarywave

from neuromorphic_materials.graph_scripts.helpers import utils, graph


def random_crop(image, crop_size, number_of_crops=25, existing_indices=None):
    """
    Randomly crops an image into smaller crops of fixed size, ensuring no repeats.

    Parameters:
        image (numpy.ndarray): Input image represented as a 2D numpy array.
        crop_size (tuple): Size of the crops in the format (rows, cols).
        number_of_crops (int): number of desired crops.
        existing_indices (list, optional): List of tuples containing already present indices.
            Defaults to None.

    Returns:
        crops (list): List of cropped images.
        top_left_indices_grid (list): List of tuples containing the indices of the top-left elements of each crop.

    """
    rows, cols = image.shape
    crop_rows, crop_cols = crop_size

    # Calculate the maximum valid starting row and column for the top-left corner of a crop
    max_start_row = rows - crop_rows
    max_start_col = cols - crop_cols

    # Initialize set of already chosen indices
    chosen_indices = set(existing_indices) if existing_indices else set()

    num_possible_crops = (max_start_row + 1) * (max_start_col + 1)
    if num_possible_crops < 10:
        raise ValueError("Image size too small to generate 10 non-repeating crops.")

    crops = []
    top_left_indices = []

    # Randomly select top-left corner indices for each crop, ensuring no repeats
    while len(crops) < number_of_crops:
        start_row = np.random.randint(0, max_start_row + 1)
        start_col = np.random.randint(0, max_start_col + 1)
        index = (start_row, start_col)

        if index not in chosen_indices:
            # Crop the image using numpy slicing
            crop = image[start_row:start_row + crop_rows, start_col:start_col + crop_cols]
            crops.append(crop)
            top_left_indices.append(index)
            chosen_indices.add(index)

    return crops, top_left_indices


def plot_dashed_crops(image_array, crop_size, output_path, ypos=0, linewidth=4, figsize=(12, 12),
                      color='red', linestyle='dashed', crop_positions=[64, 192, 320, 448]):
    '''
    Place lines to show the square cropping of the image_array

    :param image_array:
    :param crop_size:
    :param output_path:
    :param ypos:
    :param linewidth:
    :param color:
    :param linestyle:
    :param crop_positions:
    :return:
    '''
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Display the image
    ax.imshow(image_array, cmap='gray')
    # Plot dashed lines around the crops  # Horizontal positions of the crops
    for pos in crop_positions:
        rect = plt.Rectangle((pos, ypos), crop_size, crop_size, fill=False, color=color, linestyle=linestyle,
                             linewidth=linewidth)
        ax.add_patch(rect)

        # # Add dashed vertical line
        # rect_v = plt.Rectangle((0, pos), crop_size, crop_size, fill=False, color='red', linestyle='dashed',
        #                        linewidth=2)
        # ax.add_patch(rect_v)

    ax.axis('off')
    # Adjust layout
    plt.tight_layout()

    # Save the plot in SVG format
    plt.savefig(output_path, dpi=300)
    print(fr'Figure saved to: {output_path}')
    # Show the plot (optional)
    # plt.show()
    a=0

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    # print(domain)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def apply_contrast(image_gray, perc_min=2, perc_max=98):
    '''
    Apply contrast enhancement and scale values in range [0, 1]
    :param image_gray: original array
    :param perc_min:
    :param perc_max:
    :return: array after contrast and normalize between 0 and 1 is applied to it
    '''
    pixvals = image_gray
    minval = np.percentile(pixvals, perc_min)
    maxval = np.percentile(pixvals, perc_max)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1
    #plt.imshow(pixvals)
    return pixvals


def split_squares(image_gray, nrows, ncols, scale_range=None):
    l = np.array_split(image_gray, nrows, axis=0)
    new_l = []
    for a in l:
        l = np.array_split(a, ncols, axis=1)
        new_l += l

    return new_l


def crop_image_with_indexes(image, rows, cols):
    """
    Crop an image into smaller crops of size (rows, cols) and return the crops along with
    the indexes of the top-left elements of each crop.

    Parameters:
        image (numpy.ndarray): Input image array.
        rows (int): Number of rows in each crop.
        cols (int): Number of columns in each crop.

    Returns:
        cropped_images (list): List of cropped images.
        top_left_indices_grid (list): List of top-left indexes for each cropped image.
    """
    cropped_images = []
    top_left_indexes = []

    image_height, image_width = image.shape[:2]

    for r in range(0, image_height, rows):
        for c in range(0, image_width, cols):
            crop = image[r:r+rows, c:c+cols]
            cropped_images.append(crop)
            top_left_indexes.append((int(r), int(c)))

    return cropped_images, top_left_indexes


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fit', '--do_fit', type=int, default=0)
    parser.add_argument('-nr', '--nrows', type=int, default=15)
    parser.add_argument('-nc', '--ncols', type=int, default=15)
    parser.add_argument('-sp', '--save_path', type=str, default='./Dataset/GroundTruth6/')
    parser.add_argument('-lp', '--load_path', type=str, default='./JR38_CoCr_6_70000.ibw')
    parser.add_argument('-fig_form', '--fig_format', type=str, default='png')
    parser.add_argument('-wth_rand', '--wth_rand', type=int, default=0)
    args = parser.parse_args()

    # Create './Output' folder
    ensure_dir(args.save_path)
    save_name = '{:s}_ibw_'.format(args.save_path)

    # Load InPlane sample
    # r = nio.IgorIO(filename=args.load_path)
    r = binarywave.load(args.load_path)

    # current_map = np.array(r.read_analogsignal())[..., 3]
    current_map = np.array(r["wave"]["wData"])[..., 3]
    current_map = np.rot90(current_map, 1)

    # Pre-processing to produce an irregular distribution of effective conductance
    # Contrast enhancement and scaling in range [0,1]
    cmap_contrast = apply_contrast(image_gray=current_map, perc_min=2, perc_max=99)

    # sample = cmap_contrast
    sample = current_map

    # Save full sample
    s = sample
    img = 255 - (((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)
    matplotlib.image.imsave(args.save_path + 'sample.{:s}'.format(args.fig_format), img, cmap='Greys', dpi=300)
    plot_dashed_crops(image_array=sample, crop_size=128,
                      output_path=args.save_path + 'fig2a.{:s}'.format(args.fig_format),
                      # output_path=args.save_path+'sample_grid.{:s}'.format(args.fig_format),
                      crop_positions=[0, 128, 256, 384], linewidth=5, figsize=(8, 8),
                      color='yellow', linestyle='dashed')

    # Grid crops
    if args.wth_rand:
        args.save_path = args.save_path+'samples/GridLike_wthRand/'
        ensure_dir(args.save_path)
    else:
        args.save_path = args.save_path + 'samples/GridLike/'
        ensure_dir(args.save_path)

    # Crops are named with coordinates of top_left index
    # Grid crops
    segments_grid, top_left_indices_grid = crop_image_with_indexes(image=sample, rows=128, cols=128)

    # Grid crops translated by t
    t = int(sample.shape[0] / 4 // 2)
    segments_grid_translated, top_left_indices_translated = crop_image_with_indexes(image=sample[t:-t, t:-t], rows=128, cols=128)
    top_left_indices_translated = [(a + t, b + t) for (a, b) in top_left_indices_translated]

    # Add crops previously created
    segments = utils.unroll_nested_list([segments_grid, segments_grid_translated])
    top_left_indices = utils.unroll_nested_list([top_left_indices_grid, top_left_indices_translated])

    if args.wth_rand:
        # Random crops
        segments_rand, top_left_indices_rand = random_crop(sample, crop_size=(128, 128),
                                                           number_of_crops=15, existing_indices=top_left_indices_grid)
        # Add again crops previously created
        segments = utils.unroll_nested_list([segments, segments_rand])
        top_left_indices = utils.unroll_nested_list([top_left_indices, top_left_indices_rand])
        # segments = segments_rand
        # top_left_indices = top_left_indices_rand


    for i, s in enumerate(segments):
        s = 255-(((s - s.min()) / (s.max() - s.min())) * 255.9).astype(np.uint8)
        matplotlib.image.imsave(args.save_path + '{:d}_{:d}.{:s}'.format(top_left_indices[i][1], top_left_indices[i][0],
                                                                    'png'),
                                s, cmap='Greys', dpi=300)

        # matplotlib.image.imsave(args.save_path+'segm{:03d}.png'.format(i), s, cmap='Greys', dpi=300)
        # matplotlib.image.imsave(save_path_svg + '{:d}_{:d}.svg'.format(top_left_indices_grid[i][0], top_left_indices_grid[i][1]), s,
            #                         cmap='Greys')


    a=0