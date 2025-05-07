# imports
import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
from scipy import signal, sparse
import matplotlib
import matplotlib.pyplot as plt

from gen_stroke_map import gen_stroke_map
from gen_tone_map import gen_tone_map
from gen_pencil_texture import gen_pencil_texture

# It's time to pack it all up (WOOHOOO!)
def gen_pencil_drawing(img, kernel_size, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=False, w_group=0, pencil_texture_path="", stroke_darkness=1, tone_darkness=1):
    if not rgb:
        # Grayscale image:
        im = img
    else:
        # RGB image:
        yuv_img = color.rgb2yuv(img)
        im = yuv_img[:,:,0]
    # Generate the Stroke Map:
    S = gen_stroke_map(im, kernel_size, stroke_width=stroke_width, num_of_directions=num_of_directions,
                       smooth_kernel=smooth_kernel, gradient_method=gradient_method)
    S = np.power(S, stroke_darkness)
    # Generate the Tone Map:
    J = gen_tone_map(im, w_group=w_group)
    
    # Read the pencil texture:
    if not pencil_texture_path:
        pencil_texture = io.imread('./pencils/pencil0.jpg', as_gray=True)
    else:
        pencil_texture = io.imread(pencil_texture_path, as_gray=True)
    # Generate the Pencil Texture Map:
    T = gen_pencil_texture(im, pencil_texture, J)
    T = np.power(T, tone_darkness)
    # The final Y channel:
    R = np.multiply(S, T)
    
    if not rgb:
        return R
    else:
        yuv_img[:,:,0] = R
        return exposure.rescale_intensity(color.yuv2rgb(yuv_img), in_range=(0, 1))