# imports
import numpy as np
import cv2
from skimage import io, color, filters, transform, exposure
from scipy import signal, sparse
import matplotlib
import matplotlib.pyplot as plt

# Generate Stroke Map
def gen_stroke_map(img, kernel_size, stroke_width=0, num_of_directions=8, smooth_kernel="gauss", gradient_method=0):
    height = img.shape[0] # number of rows, height of the image
    width = img.shape[1] # number of columns, width of the image
    # Let's start with smoothing
    if (smooth_kernel == "gauss"):
        smooth_im = filters.gaussian(img, sigma=np.sqrt(2))
    else:
        smooth_im = filters.median(img) # default is 3x3 kernel size
    # Let's calculate the gradients:
    if not gradient_method:
        # forward gradient: (we pad with zeros)
        imX = np.zeros_like(img)
        diffX = img[: , 1:width] - img[: , 0:width - 1]
        imX[:, 0:width - 1] = diffX
        imY = np.zeros_like(img)
        diffY = img[1:height , :] - img[0:height - 1 , :]
        imY[0:height - 1, :] = diffY
        G = np.sqrt(np.square(imX) + np.square(imY))
    else:
        # Sobel
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        G = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Let's create the basic line segement (horizontal)
    # make sure it is an odd number, so the lines are at the middle
    basic_ker = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1))
    basic_ker[kernel_size + 1,:] = 1 # ------- (horizontal line)
    # Let's rotate the lines in the given directions and perform the classification:
    res_map = np.zeros((height, width, num_of_directions))
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        res_map[:,:, d] = signal.convolve2d(G, ker, mode='same')
    max_pixel_indices_map = np.argmax(res_map, axis=2)
    # What does it compute? every direction has a (height X width) matrix. For every pixel in the matrix,
    # np.argmax returns the index of the direction that holds the pixel with the maximum value
    # and thus we get the max_pixel_indices map is a (height X width) matrix with direction numbers.
    # Now we compute the Classification map:
    C = np.zeros_like(res_map)
    for d in range(num_of_directions):
        C[:,:,d] = G * (max_pixel_indices_map == d) # (max_pixel_indices_map == d) is a binary matrix
    # We should now consider the stroke width before we create S'
    if not stroke_width:
        for w in range(1, stroke_width + 1):
            if (kernel_size + 1 - w) >= 0:
                basic_ker[kernel_size + 1 - w, :] = 1
            if (kernel_size + 1 + w) < (kernel_size * 2 + 1):
                basic_ker[kernel_size + 1 + w, :] = 1
    # It's time to compute S':
    S_tag_sep = np.zeros_like(C)
    for d in range(num_of_directions):
        ker = transform.rotate(basic_ker, (d * 180) / num_of_directions)
        S_tag_sep[:,:,d] = signal.convolve2d(C[:,:,d], ker, mode='same')
    S_tag = np.sum(S_tag_sep, axis=2)
    # Remember that S shpuld be an image, thus we need to make sure the values are in [0,1]
    S_tag_normalized = (S_tag - np.min(S_tag.ravel())) / (np.max(S_tag.ravel()) - np.min(S_tag.ravel()))
    # The last step is to invert it (b->w, w->b)
    S = 1 - S_tag_normalized
    return S
    
