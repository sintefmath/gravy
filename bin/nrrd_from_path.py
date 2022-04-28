
#     *
#     * Copyright (C) 2021, SINTEF Digital, Mathematics and Cybernetics, Norway.
#     * License for use for the duration of the Change2Twin project under the terms of the Grant Agreement (No. 951956)
#     *

#
# nrrd_from_path.py
#
# Given input of a directory, create a nrrd from all png's in the dir.
# Assuming the png's width and height are all the same. Furthermore assuming
# the layers have been ordered alphanumerically.
# Also assuming the the images are binary with non-zero values meant to be 255 (white).
#
# Large models may need to reduce the quality. It is recommended to use fewer
# layers. For binary models this is achieved through averaging. For textures
# with images it is not recommended to use the average of consecutive layers,
# it is preferred to use every n'th layer. The script will automatically skip
# layers if the resulting model is larger than 1 GB (uncompressed).
#

import glob
import os
import sys
import numpy as np
from PIL import Image
import nrrd
import math
import cv2
from icecream import ic


def average_img(imgs):
    avg_img = np.mean(imgs, axis=0)
    avg_img = avg_img.astype(np.uint8)

    return avg_img

# It appears that some of the three.js calls expect the space origin and/or directions to have been set.
nrrd_header = {'space directions': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], 'space origin': [0., 0., 0.]}

# The default maximal decompressed size for the volume. WebGL 2.0 handles 2 GB,
# Chromium does not like values much above 1 GB. Use z_compression = -1 to
# lower this value to 1.0. A value of -2 will result in a size of 0.5 GB etc.
Z_COMPRESSION_LIMIT = 2.0

def nrrd_from_path(path, binary, z_compression, crop_y0, crop_y1, crop_x0, crop_x1, description_out):
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.png')]

    z_compr_limit = Z_COMPRESSION_LIMIT
    if z_compression < 0:
        compr_pow = abs(z_compression)
        compr_div = 2**compr_pow
        z_compr_limit /= compr_div
        #ic(z_compr_limit)
        z_compression = 0 # Setting auto compression.

    #ic(crop_y0, crop_y1, crop_x0, crop_x1)

    #num_avg = 1 if not binary else z_compression

    #print("len(fnames):", len(fnames))
    fnames.sort()

    # The shape returned by np is height x width. Hence a landscape 640 x 480 will be result in a 480 x 640 array.
    img_shape = np.array(Image.open(path + "/" + fnames[0])).shape
    #ic(img_shape)
    x_res = img_shape[0]
    y_res = img_shape[1]
    z_res = len(fnames)

    # Numpy swaps the axes.
    x0 = abs(crop_y0)
    x1 = crop_y1 if (crop_y1 > 0) else y_res - abs(crop_y1)
    y0 = abs(crop_x0)
    y1 = crop_x1 if (crop_x1 > 0) else x_res - abs(crop_x1)

    x_res = x1 - x0
    y_res = y1 - y0

    #ic(x_res, y_res, y0, y1, x0, x1)

    # If z_compression was set to 0 we use auto compression to stay below hardcoded limit.
    if z_compression == 0:
        z_compression = 1
        # Using 8 bits to represent a voxel (1 channel).
        number_gb = x_res*y_res*z_res/(1024.0*1024.0*1024.0) # Decompressed size.
        # If the input is not binary we do not want to take the average, then it
        # makes more sense to keep the quality and to skip layers.
        # We check if the resolution requires us 
        number_gb /= z_compression
        #ic(number_gb, binary, z_compression)
        if (number_gb > z_compr_limit):
            n_is_ok = False
            while n_is_ok == False:
                z_compression = z_compression + 1
                number_gb_n = number_gb/z_compression
                if number_gb_n < z_compr_limit:
                    n_is_ok = True
        #ic(number_gb_n, z_compression)

    skip_layers = True if (not binary and (z_compression > 1)) else False

    num_avg = z_compression if (binary) else 1

    num_images = len(fnames)
    num_layers = float(num_images)/float(z_compression)
    #ic(num_images, z_compression, num_layers, skip_layers, num_avg, binary)

    slices = []
    # We first create a slice 
    #ic(x0, x1, y0, y1)
    # We include the last layer, possibly averaging over fewer layers. This is important in order to have the same number
    # of layers for binary volume and image volume.
    num_slices = math.ceil(z_res/num_avg)#z_res
    for i in range(num_slices):
        if skip_layers and (i%z_compression != 0):
            continue
        imgs = []
        for j in range(num_avg):
            if (i*num_avg + j > num_images - 1):
                break
            img = cv2.imread(path + "/" + fnames[i*num_avg + j], 0)[y0:y1, x0:x1]
            print("i = ", i, ", img.shape:",img.shape, end='\r')
            if i < 2:
                cv2.imwrite("img.png", img)

            if binary:
                # If the input is not binary we use a threshold to make sure it is. When creating binary volume
                # from TEP images a threshold of '1' seems correct.
                th, img_thBin = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
                print("i = ", i, ", img_thBin.shape:",img_thBin.shape, end='\r')

                change_res = False
                if change_res:
                    # We may change the resolution of the images, which requires interpolating and thresholding.
                    # Using INTER_AREA appears to give a better result.
                    dim = (1600, 1600)#(xres, yres) #220, 220) # The resolution of the output images.
                    img_resized = cv2.resize(img_thBin, dim, interpolation = cv2.INTER_AREA)
                    # Removing all grey values in two steps, cutoff at the middle.
                    th, img_thBin = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)

                imgs.append(img_thBin)
            else:
                change_res = False#True
                if change_res:
                    # We may change the resolution of the images, which requires interpolating and thresholding.
                    # Using INTER_AREA appears to give a better result.
                    dim = (1600, 1600)#(2100,2100)#(636, 636) #220, 220) # The resolution of the output images.
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

                imgs.append(img)

        avg_img = average_img(imgs)

        # If the input was binary we make sure the output is also binary.
        if binary:
            th, avg_img = cv2.threshold(avg_img, 127, 255, cv2.THRESH_BINARY)

        slices.append(avg_img)

    z_res_out = len(slices)

    voxels = np.dstack(slices)

    # We fix the numpy issue with x and y axes swapped and y dir reversed.
    voxels_swapped = np.swapaxes(voxels, 0, 1)
    # Reverse the direction of y.
    voxels_swapped_flipped = np.flip(voxels_swapped, 1)

    voxels_uint8 = voxels_swapped_flipped.astype(np.uint8)

    volume_type = "binary" if binary else "images"
    filename_out = description_out + "_" + volume_type + "_" + str(x_res) + "_" + str(y_res) + "_" + str(z_res_out) + ".nrrd" 
    #ic(filename_out)

    nrrd.write(filename_out, voxels_uint8, nrrd_header)

    #print("Writing result to", filename_out)

    return [x_res, y_res, z_res_out, filename_out]

def main():

    num_arg = len(sys.argv) - 1
    if (num_arg != 8):
        print("num_arg:", num_arg)
        sys.exit("Usage: dir_name binary z_compression crop_y0 crop_y1 crop_x0 crop_x1 model_description")
        #return

    dir_name = sys.argv[1]
    binary =  int(sys.argv[2]) # z-layer compression setup: Binary: average z layers. Non-binary: Skip layers.
    z_compression = int(sys.argv[3]) # 0: auto compression, neg: auto compresssion with lower threshold. pos: Compress z-layers by n.
    crop_y0 = int(sys.argv[4]) # y direction (vertical, from top to bottom)
    crop_y1 = int(sys.argv[5]) # Note that this value should be negative if counting from the ymax value.
    crop_x0 = int(sys.argv[6]) # y direction (horizontal, from left to right)
    crop_x1 = int(sys.argv[7]) # Note that this value should be negative if counting from the xmax value.
    description_out = sys.argv[8]

    # if z_compression < 0:
    #     print("z_compression must be a non-negative integer, not", z_compression, ", setting it to 0 (auto compression).")
    #     z_compression = 0

    # Stripping any "/" from the dir_name.
    dir_name = os.path.abspath(dir_name)

    nrrd_from_path(dir_name, binary, z_compression, crop_y0, crop_y1, crop_x0, crop_x1, description_out)

if __name__ == '__main__':
    num_arg = len(sys.argv) - 1
    if (num_arg != 8):
        print("num_arg:", num_arg)
        sys.exit("Usage: dir_name binary z_compression crop_y0 crop_y1 crop_x0 crop_x1 model_description")
        #return
    main()
