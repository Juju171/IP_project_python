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
from gen_pencil_drawing import gen_pencil_drawing


# # Let's see an example Stroke Map:


# ex_img = io.imread('./inputs/7--129.jpg')
# ex_img_yuv = color.rgb2yuv(ex_img)
# ex_img_y_ch = ex_img_yuv[:,:,0]
# ex_img_stroke_map = gen_stroke_map(ex_img_y_ch, kernel_size=8, stroke_width=1,
#                                    num_of_directions=8, smooth_kernel="gauss", gradient_method=1)
# plt.rcParams['figure.figsize'] = [16, 8]
# plt.subplot(1,2,1)
# plt.imshow(ex_img_stroke_map, cmap='gray')
# plt.axis('off')
# plt.title('Stroke Map')
# plt.subplot(1,2,2)
# plt.imshow(ex_img)
# plt.axis('off')
# plt.title('Original Image')
# plt.show()




# # Let's see an example Tone Map:


# ex_img = io.imread('./inputs/2--31.jpg')
# ex_img_yuv = color.rgb2yuv(ex_img)
# ex_img_y_ch = ex_img_yuv[:,:,0]
# ex_img_tone_map_0 = gen_tone_map(ex_img_y_ch, w_group=0)
# ex_img_tone_map_1 = gen_tone_map(ex_img_y_ch, w_group=1)
# ex_img_tone_map_2 = gen_tone_map(ex_img_y_ch, w_group=2)
# plt.rcParams['figure.figsize'] = [16, 8]
# plt.subplot(1,4,1)
# plt.imshow(ex_img_y_ch, cmap='gray')
# plt.axis('off')
# plt.title('Original Image (Gray)')
# plt.subplot(1,4,2)
# plt.imshow(ex_img_tone_map_0, cmap='gray')
# plt.axis('off')
# plt.title('Tone Map - Group 1')
# plt.subplot(1,4,3)
# plt.imshow(ex_img_tone_map_1, cmap='gray')
# plt.axis('off')
# plt.title('Tone Map - Group 2')
# plt.subplot(1,4,4)
# plt.imshow(ex_img_tone_map_2, cmap='gray')
# plt.axis('off')
# plt.title('Tone Map - Group 3')

# plt.show()



# Let's see an example Pencil Texture Map:


# ex_img = io.imread('./inputs/3--17.jpg')
# ex_img_yuv = color.rgb2yuv(ex_img)
# ex_img_y_ch = ex_img_yuv[:,:,0]
# pencil_tex = io.imread('./pencils/pencil0.jpg', as_gray=True)
# ex_img_tone_map_0 = gen_tone_map(ex_img_y_ch, w_group=0)
# ex_img_tex_map = gen_pencil_texture(ex_img_y_ch, pencil_tex, ex_img_tone_map_0)

# plt.rcParams['figure.figsize'] = [16, 8]
# plt.subplot(1,2,1)
# plt.imshow(ex_img_tex_map, cmap='gray')
# plt.axis('off')
# plt.title('Final Texture Map')
# plt.subplot(1,2,2)
# plt.imshow(ex_img)
# plt.axis('off')
# plt.title('Original Image')
# plt.show()




# Let's see some results


plt.subplot(1,2,1)
ex_img = io.imread('./inputs/3--17.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=1, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=1, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                               stroke_darkness= 2,tone_darkness=1.5)
plt.imshow(ex_img)
plt.axis("off")
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(ex_im_pen)
plt.axis("off")
plt.title("Pencil Drawing")
plt.show()

plt.subplot(1,2,1)
ex_img = io.imread('./inputs/2--83.jpg')
pencil_tex = './pencils/pencil0.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=2, num_of_directions=8, smooth_kernel="median",
                       gradient_method=0, rgb=True, w_group=0, pencil_texture_path=pencil_tex,
                               stroke_darkness= 2,tone_darkness=1.5)
plt.imshow(ex_img)
plt.axis("off")
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(ex_im_pen)
plt.axis("off")
plt.title("Pencil Drawing")
plt.show()

plt.subplot(1,2,1)
ex_img = io.imread('./inputs/17--8.jpg')
pencil_tex = './pencils/pencil3.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=1, rgb=True, w_group=1, pencil_texture_path=pencil_tex,
                               stroke_darkness= 2,tone_darkness=1.5)
plt.imshow(ex_img)
plt.axis("off")
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(ex_im_pen)
plt.axis("off")
plt.title("Pencil Drawing")
plt.show()