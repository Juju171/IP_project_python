# imports
import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
from scipy import signal, sparse
import matplotlib
import matplotlib.pyplot as plt

# Generate Tone Map
# Make sure input image is in [0,1]
def gen_tone_map(img, w_group=0):
    # The first thing we need to do is to calculate the parameters and define weight groups
    w_mat = np.array([[11, 37, 52],
                     [29, 29, 42],
                     [2, 22, 76]])
    w = w_mat[w_group,:]
    # We can now define tone levels like:
    # dark: [0-85]
    # mild: [86-170]
    # bright: [171-255]
    # Assign each pixel a tone level, make 3 lists where each list holds the pixels (indices) of every tone.
    # Use these lists to calculate the parameters for each image.
    
    # For simplicity, we will use the parameters from the paper:
    # for the mild layer:
    u_b = 225
    u_a = 105
    # for the bright layer:
    sigma_b = 9
    # for the dark layer:
    mu_d = 90
    sigma_d = 11
    
    # Let's calculate the new histogram (p(v)):
    num_pixel_vals = 256
    p = np.zeros(num_pixel_vals)
    for v in range(num_pixel_vals):
        p1 = (1 / sigma_b) * np.exp(-(255 - v) / sigma_b)
        if (u_a <= v <= u_b):
            p2 = 1 / (u_b - u_a)
        else:
            p2 = 0
        p3 = (1 / np.sqrt(2 * np.pi * sigma_d)) * np.exp( (-np.square(v - mu_d)) / (2 * np.square(sigma_d)) )
        p[v] = w[0] * p1 + w[1] * p2 + w[2] * p3 * 0.01
    # normalize the histogram:
    p_normalized = p / np.sum(p)
    # calculate the CDF of the desired histogram:
    P = np.cumsum(p_normalized)
    # calculate the original histogram:
    h = exposure.histogram(img, nbins=256)
    # CDF of original:
    H = np.cumsum(h / np.sum(h))
    # histogram matching:
    lut = np.zeros_like(p)
    for v in range(num_pixel_vals):
        # find the closest value:
        dist = np.abs(P - H[v])
        argmin_dist = np.argmin(dist)
        lut[v] = argmin_dist
    lut_normalized = lut / num_pixel_vals
    J = lut_normalized[(255 * img).astype(int)]
    # smooth:
    J_smoothed = filters.gaussian(J, sigma=np.sqrt(2))
    return J_smoothed
    
