from enum import Enum
from pprint import pprint

import cv2
import imutils
import numpy as np
from .polar_plot import polar_plot
from matplotlib import pyplot as plt
from scipy.ndimage import convolve, gaussian_filter, gaussian_filter1d, rotate


class EdgeDetectionMethod(Enum):
    SOBEL = "sobel"
    GAUSSIAN = "gaussian"
    GAUSSIAN_ROTATE_IMG = "gaussian_rotate_img"


def extract_angles(conv_x: np.ndarray, conv_y: np.ndarray, omit_zero: bool = True):
    """Return extracted angles in interval (0, 360]"""
    # Set near-zero values to zero
    conv_x = np.where((conv_x < -1e-10) | (conv_x > 1e-10), conv_x, 0)
    conv_y = np.where((conv_y < -1e-10) | (conv_y > 1e-10), conv_y, 0)
    # Omit pixels where both x and y gradients are zero, to avoid large zero angle spike
    if omit_zero:
        indices = (conv_x != 0) | (conv_y != 0)
        conv_x = conv_x[indices]
        conv_y = conv_y[indices]
    else:
        # All indices will be returned if zeros aren't omitted
        indices = np.full_like(conv_x, True, dtype=bool)

    # Calculate the direction of the gradient in degrees, and return indices too
    angles = np.rad2deg(np.arctan2(conv_y, conv_x)) + 180
    if angles.size == 0:
        print("Zero angles extracted, consider checking input image")
    return angles, indices


def extract_pixel_angles_sobel(img):
    # Returns array of the size of number of pixels of the image.
    # where in each element there is the value of the detected angle in each pixel.
    # Detected angles are in DEGREES between (-180, +180]
    img = img.squeeze()
    # img = np.flipud(img)
    # sobel_x = ndimage.sobel(img, axis=0, mode='constant')
    # sobel_y = ndimage.sobel(img, axis=1, mode='constant')
    # One angle for each pixel
    sobel_kernel = 3
    # 2) Take the gradient in x and y separately
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 1, 0, sobel_kernel)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, sobel_kernel)
    # 3) Calculate the direction of the gradient
    return extract_angles(sobel_x, sobel_y)


def gaussian_kernel_x_y(kernel_size: int, sigma: float):
    # Generate a kernel that is 2 larger than the requested size
    x = cv2.getGaussianKernel(kernel_size + 2, sigma)
    y = cv2.getGaussianKernel(kernel_size + 2, sigma)
    kernel = x.dot(y.T)
    # Compute the gradient using larger kernel, and return
    # the requested kernel size from the resulting gradients
    return np.array(np.gradient(kernel))[:, 1:-1, 1:-1]


def extract_pixel_angles_gaussian(img: np.ndarray, kernel_size: int, sigma: float):
    # Returns array of the size of number of pixels of the image.
    # where in each element there is the value of the detected angle in each pixel.
    img = img.squeeze()
    img = img.astype(float)
    kernel_x, kernel_y = gaussian_kernel_x_y(kernel_size, sigma)
    conv_x = convolve(img, kernel_x, mode="nearest")
    conv_y = convolve(img, kernel_y, mode="nearest")
    return extract_angles(conv_x, conv_y)


def extract_angles_gaussian_45_deg_eval(
    img: np.ndarray, kernel_size: int, sigma: float, margin: float
):
    """Compute empirical PDF by rotating image while keeping evaluation angle static"""
    img = img.squeeze()
    img = img.astype(float)
    kernel_x, kernel_y = gaussian_kernel_x_y(kernel_size, sigma)
    empirical_pdf = np.empty(360)
    # Rotate the image with an increasing angle
    for angle in range(90):
        rot_img = imutils.rotate_bound(img, angle=angle - 45)
        conv_x = convolve(rot_img, kernel_x, mode="nearest")
        conv_y = convolve(rot_img, kernel_y, mode="nearest")
        # Extract angles from the Gaussian derivative-convolved images
        angles = extract_angles(conv_x, conv_y)[0]

        # Skip if no angles were extracted at all
        if angles.size == 0:
            for quadrant in range(0, 360, 90):
                empirical_pdf[quadrant + angle] = 0
            continue

        # Set the output for each 45-degree angle as the fraction of all extracted
        # angles in this rotation, that are within a margin of each 45-degree angle
        for quadrant in range(0, 360, 90):
            empirical_pdf[quadrant + angle] = (
                (angles >= quadrant + 45 - margin) & (angles <= quadrant + 45 + margin)
            ).sum() / angles.size

    # Normalise output to ensure it is a PDF
    return empirical_pdf / empirical_pdf.sum()
