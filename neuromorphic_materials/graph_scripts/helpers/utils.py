#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from skimage import io
from skimage.color import gray2rgb, rgb2gray

def unroll_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def min_max_order_of_magnitude(numbers):
    import math
    min_order = math.inf
    max_order = -math.inf

    for num in numbers:
        if num==0:
            pass
        else:
            order = int(math.log10(abs(num)))
            min_order = min(min_order, order)
            max_order = max(max_order, order)

    return min_order, max_order

def pickle_save(filename, obj, protocol=4, plural=False):
    if plural:
        pickle.dump(obj, open(filename, "wb"), protocol=protocol)
    else:
        pickle.dump(obj, open(filename, "wb"), protocol=protocol)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def apply_clahe_to_grey_arr(arr, cliplim=8, tileGrideSize=(8, 8)):
    print(
        "Apply CLAHE for contrast enhancement: cliplim ",
        cliplim,
        ", tileGrideSize ",
        tileGrideSize,
    )
    clahe = cv2.createCLAHE(clipLimit=cliplim, tileGridSize=tileGrideSize)
    arr = clahe.apply(arr)
    return arr


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    # print(domain)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


# Function to calculate Chi-distance
def chi2_distance(a, b):
    # compute the chi-squared distance using above formula
    a = a + np.finfo(float).eps
    b = b + np.finfo(float).eps
    # chi =  np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(A, B)])
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(a, b)])
    return chi


def empirical_pdf_and_cdf(sample, bins=100):
    # if edges is None:
    (count_c, bins_c) = np.histogram(sample, bins=bins)
    # else: count_c, bins_c, = np.histogram(sample, bins=edges)
    # define the empirical pdf
    my_pdf = count_c / np.sum(count_c)
    # define the empirical cdf
    my_cdf = np.zeros_like(bins_c)
    my_cdf[1:] = np.cumsum(my_pdf)
    return my_pdf, my_cdf, bins_c


def bilinear_resize_vectorized(image, new_shape):
    """
    `image` is a 2-D numpy array
    `height` and `width` are the desired spatial dimension of the new 2-D array.
    """
    height = new_shape[0]
    width = new_shape[1]
    img_height, img_width = image.shape

    image = image.ravel()

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    y, x = np.divmod(np.arange(height * width), width)

    x_l = np.floor(x_ratio * x).astype("int32")
    y_l = np.floor(y_ratio * y).astype("int32")

    x_h = np.ceil(x_ratio * x).astype("int32")
    y_h = np.ceil(y_ratio * y).astype("int32")

    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = image[y_l * img_width + x_l]
    b = image[y_l * img_width + x_h]
    c = image[y_h * img_width + x_l]
    d = image[y_h * img_width + x_h]

    resized = (
        a * (1 - x_weight) * (1 - y_weight)
        + b * x_weight * (1 - y_weight)
        + c * y_weight * (1 - x_weight)
        + d * x_weight * y_weight
    )

    return resized.reshape(height, width)


def lzw_complexity_img(intensity_array):
    rows = intensity_array.shape[0]
    cols = intensity_array.shape[1]

    int_string = np.zeros((rows * cols))
    idx = 0
    # Creating a string of all intensity values
    for i in range(0, rows):
        for j in range(0, cols):
            int_string[idx] = intensity_array[i, j]
            idx = idx + 1

    # print(int_string)
    crs = ""  # currently recognized sequence
    curr = ""  # current sequence

    output = {}
    out_idx = 0

    dict_val = {}
    dict_idx = 0

    for i in range(0, 255 + 1):
        dict_val[str(i)] = i
    # print(dict_val)
    # print(len(dict_val))
    # print(dict_val[255])
    # next unused location
    dict_idx = 256 + 1

    curr = int_string[0]

    crs = str(int(curr))

    for i in range(1, idx):
        if i % (idx / 4) == 0:
            pass  # print(i,'/',idx)
        curr = int_string[i]

        t_str = crs + "-" + str(int(curr))

        # print("t_str is " + t_str)

        if t_str in dict_val:
            # print(t_str + " Already exists");
            crs = t_str
        else:
            # if not found in the dictionary
            # print("Creating a new entry for the dictionary ")
            # print(crs)
            output[out_idx] = dict_val[crs]
            # print("Output " , + output[int(out_idx)])
            out_idx = out_idx + 1
            crs = str(int(curr))

            # add the new entry to the dictionary
            dict_val[t_str] = dict_idx
            # print(dict_val)
            dict_idx = dict_idx + 1

    # Last entry will always be found in the dictionary
    if crs in dict_val:
        # print(crs)
        output[out_idx] = dict_val[crs]
        # print("Output " , + output[int(out_idx)])
        out_idx = out_idx + 1

    # printing the encoded output
    # print(output.values());
    string = "\nLZW ratio\n%d/%d=%f" % (
        len(output),
        len(int_string),
        len(output) / len(int_string),
    )
    string = "\nLZW ratio=%f" % (len(output) / len(int_string))
    # print(string)
    return len(output) / len(int_string)
    # return((len(output), len(int_string),len(output)/len(int_string)))


def to_gray_uint(image):
    return np.uint8(rgb2gray(image) * 255)


def load_image(load_path: Path):
    # load image
    image = io.imread(str(load_path.resolve()))

    print("\nImage shape: ", image.shape, "\n")
    if len(image.shape) == 2:
        return image, gray2rgb(image)
    elif len(image.shape) > 2:
        image_rgb = image[..., :-1] if image.shape[2] > 3 else image
        return to_gray_uint(image_rgb), image_rgb
    else:
        return None, None
