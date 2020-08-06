import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import random
import shutil
import re
#from scipy.misc import imresize
#from mammo_utils import create_mask, half_image, get_fuzzy_offset, progress, rename_and_copy_files

## for the masks that are too big we are going to attempt to slice it into four slices, one for each corner.
## this function makes sure that the corners don't run off the image
def check_slice_corners(image_slice, full_image_arr, corners, slice_size=299):
    h, w = image_slice.shape
    image_h, image_w = full_image_arr.shape

    # if the slice is the right shape return it
    if (h == slice_size) & (w == slice_size) and (np.mean(image_slice) > 25):
        return image_slice

    # else try to reframe it by checking each corner
    else:
        if corners[0] < 0:
            corners[0] = 0
            corners[1] = slice_size
        elif corners[1] > image_h:
            corners[1] = image_h
            corners[0] = image_h - slice_size

        if corners[2] < 0:
            corners[2] = 0
            corners[3] = slice_size
        elif corners[3] > image_w:
            corners[3] = image_w
            corners[2] = image_w - slice_size

        image_slice = full_image_arr[corners[0]:corners[1], corners[2]:corners[3]]

        # check that the slice contains useable info and doesn't just contain empty space
        if (np.mean(image_slice) > 25):
            return image_slice

    # else there is a problem, return an unusable array
    return np.array([0]).reshape((1, 1))


## remove extraneous characters from end of file name and return it
def clean_name(name):
    patient_id = re.findall("(P_[\d]+)_", name)
    if len(patient_id) > 0:
        patient_id = patient_id[0]
    else:
        print("Name error")
        return name

    image_side = re.findall("_(LEFT|RIGHT)_", name)

    if len(image_side) > 0:
        image_side = image_side[0]
    else:
        print("Side error")
        return name

    image_type = re.findall("(CC|MLO)", name)
    if len(image_type) > 0:
        image_type = image_type[0]
    else:
        return name

    return patient_id + "_" + image_side + "_" + image_type


## randomly flip an image left-right, up-down or both and return it
def random_flip_image(img):
    fliplr = np.random.binomial(1, 0.5)
    flipud = np.random.binomial(1, 0.5)

    if fliplr:
        img = np.flip(img, 1)
    if flipud:
        img = np.flip(img, 0)

    return img


def get_roi_edges(center_col, center_row, img_height, img_width, fuzz_offset_w=0, fuzz_offset_h=0, scale_factor=1,
                  slice_size=299):
    # slice margin
    slice_margin = slice_size // 2

    # figure out the new center of the ROI
    center_col_scaled = int(center_col * scale_factor)
    center_row_scaled = int(center_row * scale_factor)

    start_col = center_col_scaled - slice_margin + fuzz_offset_h
    end_col = start_col + slice_size

    if start_col < 0:
        start_col = 0
        end_col = slice_size
    elif end_col > img_width:
        end_col = img_width
        start_col = img_width - slice_size

    start_row = center_row_scaled - slice_margin + fuzz_offset_w
    end_row = start_row + slice_size

    if start_row < 0:
        start_row = 0
        end_row = slice_size
    elif end_row > img_height:
        end_row = img_height
        start_row = img_height - slice_size

    return start_row, end_row, start_col, end_col


## function to create ROI slices from masks and full images
## inputs: mask_dir and image_dir - paths to directories containing respective images
##         slice_size - size of slices to create
##         output - if True will output each file processed,
##                  if False will only output errors and warnings,
##                  if None won't ouput anything
## returns: numpy array of images and list of corresponding file names
def create_roi_slices(mask_dir, image_dir, slice_size=299, output=True):
    # loop through mask directory
    mask_files = os.listdir(mask_dir)
    slices_list = []
    image_names_list = []
    full_slice_size = slice_size * 2

    counter = 0
    if output is None:
        progress(counter, len(mask_files), 'WORKING')

    for mask in mask_files:
        # create a progress bar
        counter += 1
        if output is None:
            progress(counter, len(mask_files), mask)

        if output:
            print(mask)

        # get the base file name from the mask name
        base_image_file = clean_name(mask)
        image_file = base_image_file + "_FULL.jpg"

        # try to open the image, if it doesn't exist continue
        try:
            full_image = PIL.Image.open(image_dir + "/" + image_file)
        except:
            try:
                full_image = PIL.Image.open(image_dir + "/" + base_image_file + "000000.jpg")
            except:
                try:
                    full_image = PIL.Image.open(image_dir + "/" + base_image_file + "000001.jpg")
                except:
                    print("Error FileNotFound:", base_image_file)
                    continue

                    # scale the full image to half size
        # full_image = half_image(full_image)

        # turn it into an array
        full_image_arr = np.array(full_image)

        # get rid of the extra dimensions with unneccessary info
        full_image_arr = full_image_arr[:, :, 0]

        # get the mask, if the mask is bigger than the slice we will create multiple slices using the corners of the mask.
        center_row, center_col, too_big, full_image_arr, mask_size = create_mask(mask_dir + "/" + mask, full_image_arr,
                                                                                 half=False, output=output)

        # get some info we will need later
        image_h, image_w = full_image_arr.shape
        try:
            mask_height = mask_size[0]
            mask_width = mask_size[1]
            roi_size = np.max([mask_height, mask_width])
            if output:
                print("Mask", mask, " Height:", mask_height, "Width:", mask_width)
        except:
            print("Mask Size Error:", mask_size, "for", mask)

        # if there is a problem then skip this one
        if (center_row == 0) and (center_col == 0):
            print("Error, skipping", mask)
            continue

        ## add the ROI at normal size no matter what
        if True:
            if output:
                print("Adding ROI at size", mask)

            # if the ROI is smaller than the slice do some offsetting, otherwise leave it as is
            if roi_size <= (full_slice_size - 60):
                fuzz_offset_h, fuzz_offset_w = get_fuzzy_offset(roi_size + 10, slice_size=full_slice_size)
            else:
                fuzz_offset_h, fuzz_offset_w = 0, 0

            # define boundaries for the abnormality
            start_row, end_row, start_col, end_col = get_roi_edges(center_col, center_row, image_h, image_w,
                                                                   fuzz_offset_w, fuzz_offset_h, 1,
                                                                   slice_size=full_slice_size)

            # slice the ROI out of the image
            img_slice = full_image_arr[start_row:end_row, start_col:end_col]

            # cut the slice down to half size
            img_slice = imresize(img_slice, (slice_size, slice_size))

            # if the slice is the right shape add it to the list
            if img_slice.shape == (slice_size, slice_size):
                slices_list.append(img_slice.reshape(slice_size, slice_size, 1))
                image_names_list.append(base_image_file + ".jpg")
            elif output is not None:
                print("Normal slice is wrong shape", mask)

        ## if the ROI is either too big or too small cut out the full ROI and zoom it to size
        if too_big or (roi_size <= full_slice_size // 1.5):
            if output is not None:
                print("Adding zoomed ROI", mask)

            # Add a 20% margin to the ROI area
            roi_margin = int(roi_size * 0.20)
            roi_size_w_margin = roi_size + roi_margin

            # set a lower bound to the zoom so we don't lose too much info by zooming in, 1.5x zoom seems a good max
            if roi_size_w_margin <= 400:
                zoom_roi_size_w_margin = 400
            else:
                zoom_roi_size_w_margin = roi_size_w_margin

            # define a random offset so the images are not all
            fuzz_offset_h, fuzz_offset_w = get_fuzzy_offset(roi_size, slice_size=zoom_roi_size_w_margin)

            # define boundaries for the ROI
            start_row, end_row, start_col, end_col = get_roi_edges(center_col, center_row, image_h, image_w,
                                                                   fuzz_offset_w, fuzz_offset_h, 1,
                                                                   slice_size=zoom_roi_size_w_margin)

            # slice the ROI out of the image
            img_slice = full_image_arr[start_row:end_row, start_col:end_col]

            # make sure the slice is the right shape
            if img_slice.shape[0] == img_slice.shape[1]:
                # resize it to 299x299 and add it to the list
                img_slice = random_flip_image(imresize(img_slice, (slice_size, slice_size)))
                slices_list.append(img_slice.reshape(slice_size, slice_size, 1))
                image_names_list.append(base_image_file + ".jpg")
            else:
                print("Resized slice is wrong shape", mask)

        ## if the ROI is bigger than a slice on either side, we will cut it out as tiles at regular size
        if roi_size > 598:
            if output is not None:
                print("Cutting ROI into tiles", mask)

            # first define the corners of the ROI, making sure the ROI is at least as big as a slice should be
            if mask_height >= 598:
                first_row = center_row - (mask_height // 2)
                last_row = center_row + (mask_height // 2)
            else:
                first_row = center_row - 299
                last_row = first_row + 598

            if mask_width >= 598:
                first_col = center_col - (mask_width // 2)
                last_col = center_col + (mask_width // 2)
            else:
                first_col = center_col - 299
                last_col = first_col + 298

            # cut out the full ROI
            roi_img = full_image_arr[first_row:last_row, first_col:last_col]

            # cut the ROI into tiles of size 598x598, with a stride of 299
            tiles = [roi_img[x:x + full_slice_size, y:y + full_slice_size] for x in
                     range(0, roi_img.shape[0], slice_size) for y in range(0, roi_img.shape[1], slice_size)]

            # loop through the tiles, if they are the right size size them down and add them to the list
            for tile in tiles:
                # if the tile is square
                if tile.shape[0] == tile.shape[1]:
                    # if the tile has information in it
                    if np.mean(tile) >= 23:
                        img_slice = random_flip_image(imresize(tile, (slice_size, slice_size)))
                        slices_list.append(img_slice.reshape(slice_size, slice_size, 1))
                        image_names_list.append(base_image_file + ".jpg")

    return np.array(slices_list), image_names_list