#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Dec 6 18:24:00 2019

@author: Guillaume Théroux-Rancourt and Matt Jenkins
"""


import sys
import numpy as np
import os
from pandas import DataFrame
from skimage import transform, img_as_bool, img_as_int, img_as_ubyte
import skimage.io as io
from skimage.measure import label, marching_cubes_lewiner, mesh_surface_area, regionprops, marching_cubes_classic
# import zipfile
import gc
from Leaf_Segmentation_Functions_py3 import delete_dangling_epidermis, openAndReadFile, Trim_Individual_Stack, define_params_traits
#import cv2

__author__ = "Guillaume Théroux-Rancourt and Matt Jenkins"
__copyright__ = ""
__credits__ = ["Guillaume Théroux-Rancourt and Matt Jenkins"]
__license__ = "MIT"
__version__ = "0.2.1"
__maintainer__ = "Guillaume Théroux-Rancourt"
__email__ = "guillaume.theroux-rancourt@boku.ac.at"
__status__ = "beta"

# Last edited by: MJ on Dec 06, 2019 18:24:00

def main():
    # Extract data from command line input
    path = sys.argv[0]

    # create python-dictionary from command line inputs (ignore first)
    sys.argv = sys.argv[1:]
    arg_dict = dict(j.split('=') for j in sys.argv)

    # define variables for later on
    filenames = [] # list of arg files
    req_not_def = 0  # permission variable for required definitions when using command line option

    # define important variables using command line
    for key, value in arg_dict.items():
        if key == 'argfiles':
            for z in value.split(','):
                z.strip()
                z = z.replace('\n','')
                filenames.append(z)
        if key == 'path_to_argfile_folder':
            path_to_argfile_folder = str(value)
        else:
            # read in desired values for parameters
            if key == 'sample_name':
                sample_name = str(value)
            if key == 'px_size':
                px_edge = float(value)
            if key == 'units':
                units = str(value)
            if key == 'rescale_factor':
                to_resize = int(value)
            if key == 'reuse_binary':
                reuse_raw_binary = str(value)
            # Binary suffix is only required if reuse binary is set to True
            if key == 'binary_suffix':
                binary_postfix = str(value)
            if key == 'trim_slices':
                trim_slices = int(value)
            if key == 'trim_column_L':
                trim_column_L = int(value)
            if key == 'trim_column_R':
                trim_column_R = int(value)
            if key == 'tissue_values':
                color_values = str(value)
            if key == 'path_to_image_folder':
                base_folder_name = str(value)
            # set up default values for some optional parameters
            try: trim_slices
            except NameError: trim_slices = 0
            try: trim_column_L
            except NameError: trim_column_L = 0
            try: trim_column_R
            except NameError: trim_column_R = 0
            try: px_edge
            except NameError: px_edge = 1


    if len(filenames)>0:
        # define some things
        j = 0
        permission = 0

        for i in range(0,len(filenames)): # optional but nice catch for incorrect filepath or filename entry
            if os.path.exists(path_to_argfile_folder+filenames[i]) == False:
                print("\nThe argument file is not present in the arg_file folder or path_to_argfile_folder is incorrect. Try again.\n")
                permission = 1

        while j < len(filenames) and permission == 0:
            print('\nWorking on scan: '+str(j+1)+' of '+str(len(filenames))+'\n')

            #read input file and define lots of stuff
            list_of_lines = openAndReadFile(path_to_argfile_folder+filenames[j])
            # print(list_of_lines) # comment out once built

            # define parameters using list_of_lines
            sample_name, binary_postfix, px_edge, units, to_resize, reuse_raw_binary, trim_slices, trim_column_L, trim_column_R, color_values, base_folder_name = define_params_traits(list_of_lines)

            # Pixel dimmension
            vx_volume = px_edge**3
        # TESTING
            # print(vx_volume)
            # print(color_values)
        # TESTING

            # Define the different tissue values
            epid_value, bg_value, mesophyll_value, ias_value, vein_value, bs_value = [int(x) for x in color_values.split(',')]

        # TESTING
            # print(epid_value, bg_value, mesophyll_value)
        # TESTING

            # Load segmented image
            # Set directory of functions in order to import MLmicroCTfunctions
            # path_to_script = '/'.join(path.split('/')[:-1]) + '/'
            # os.chdir(path_to_script)

            sample_path_split = sample_name.split('/')
        # TESTING
                # print(sample_path_split)
        # TESTING

            # If input path to sample is of length 1, i.e. only the sample name,
            # create the folder names based on default file naming.
            if len(sample_path_split) == 1:
                folder_name = '/MLresults/'
                raw_ML_prediction_name = sample_name + 'fullstack_prediction.tif'
            else:
                sample_name = sample_path_split[-3]
                folder_name = sample_path_split[-2] + '/'
                raw_ML_prediction_name = sample_path_split[-1]
            filepath = base_folder_name + sample_name + '/'
            if reuse_raw_binary == 'True':
                binary_filename = sample_name + binary_postfix
            # raw_ML_prediction_name = sample_name + 'fullstack_prediction.tif'

            print('')
            print('#################')
            print('# STARTING WITH #')
            print('#################')
            print(' ' + sample_name)

        #
            # Check if the file has already been processed -- Just in case!
            if os.path.isfile(filepath + sample_name + 'RESULTS.txt'):
                print('')
                print('This file has already been processed!')
                print('')
                assert False

            if os.path.isfile(base_folder_name + sample_name + '/' + sample_name + 'SEGMENTED.tif'):
                print('###LOADING POST-PROCESSED SEGMENTED STACK###')
                large_segmented_stack = io.imread(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif')
            else:
                # Load the ML segmented stack
                raw_pred_stack = io.imread(filepath + folder_name + raw_ML_prediction_name)
                uniq100th = np.unique(raw_pred_stack[100])

                if np.any(uniq100th < 0):
                    raw_pred_stack = np.where(raw_pred_stack < 0, raw_pred_stack + 256, raw_pred_stack)
                    print(np.unique(raw_pred_stack[100]))
                else:
                    print(uniq100th)

                # Trim at the edges -- The ML does a bad job there
                if trim_slices == 0:
                    if trim_column_L == 0:
                        if trim_column_R == 0:
                            raw_pred_stack = raw_pred_stack
                else:
                    if trim_column_L == 0:
                        if trim_column_R == 0:
                            raw_pred_stack = raw_pred_stack = raw_pred_stack[trim_slices:-trim_slices, :, :]
                    else:
                        raw_pred_stack = raw_pred_stack[trim_slices:-trim_slices, :, trim_column_L:-trim_column_R]
    #
    #
                ###################
                # EPIDERMIS
                ###################
                print('')
                print('### EPIDERMIS ###')
                print('')

                # Label all of the epidermis regions
                unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
                props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

                # io.imshow(unique_epidermis_volumes[100])

                # Find the size and properties of the epidermis regions
                epidermis_area = np.zeros(len(props_of_unique_epidermis))
                epidermis_label = np.zeros(len(props_of_unique_epidermis))
                epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                for regions in np.arange(len(props_of_unique_epidermis)):
                    epidermis_area[regions] = props_of_unique_epidermis[regions].area
                    epidermis_label[regions] = props_of_unique_epidermis[regions].label
                    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                # Find the two largest epidermis
                ordered_epidermis = np.argsort(epidermis_area)
                print('The two largest values below should be in the same order of magnitude')
                print((epidermis_area[ordered_epidermis[-4:]]))

                if epidermis_area[ordered_epidermis[-1]] > (10*epidermis_area[ordered_epidermis[-2]]):
                    print('ERROR: Both epidermis might be connected!')
                    print('Trying to remove the dangling epidermis pixels')
                    no_dangling = delete_dangling_epidermis((unique_epidermis_volumes == ordered_epidermis[-1]+1), True, False)

                    # Label all of the epidermis regions
                    unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
                    props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

                    # io.imshow(unique_epidermis_volumes[100])

                    # Find the size and properties of the epidermis regions
                    epidermis_area = np.zeros(len(props_of_unique_epidermis))
                    epidermis_label = np.zeros(len(props_of_unique_epidermis))
                    epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                    for regions in np.arange(len(props_of_unique_epidermis)):
                        epidermis_area[regions] = props_of_unique_epidermis[regions].area
                        epidermis_label[regions] = props_of_unique_epidermis[regions].label
                        epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                    # Find the two largest epidermis
                    ordered_epidermis = np.argsort(epidermis_area)
                    print('The two largest values below should be in the same order of magnitude')
                    print((epidermis_area[ordered_epidermis[-4:]]))

                    if epidermis_area[ordered_epidermis[-1]] > (10*epidermis_area[ordered_epidermis[-2]]):
                        print('#########################################')
                        print('#########################################')
                        print('ERROR: Both epidermis are still connected!')
                        print('' + sample_name)
                        print('#########################################')
                        print('#########################################')
                        assert False

                print("")
                print('The center of the epidermis should be more or less the same on the')
                print('1st and 3rd columns for the two largest values.')
                print((epidermis_centroid[ordered_epidermis[-2:]]))
                print("")

                two_largest_epidermis = (unique_epidermis_volumes
                                         == ordered_epidermis[-1]+1) | (unique_epidermis_volumes == ordered_epidermis[-2]+1)

                #Check if it's correct
                #io.imsave(filepath + folder_name + 'test_epidermis.tif',
                #          img_as_ubyte(two_largest_epidermis))
                # io.imshow(two_largest_epidermis[100])


                # Get the values again: makes it cleaner
                unique_epidermis_volumes = label(two_largest_epidermis, connectivity=1)
                props_of_unique_epidermis = regionprops(unique_epidermis_volumes)
                epidermis_area = np.zeros(len(props_of_unique_epidermis))
                epidermis_label = np.zeros(len(props_of_unique_epidermis))
                epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                for regions in np.arange(len(props_of_unique_epidermis)):
                    epidermis_area[regions] = props_of_unique_epidermis[regions].area
                    epidermis_label[regions] = props_of_unique_epidermis[regions].label
                    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                ## io.imshow(unique_epidermis_volumes[100])

                # Transform the array to 8-bit: no need for the extra precision as there are only 3 values
                unique_epidermis_volumes = np.array(unique_epidermis_volumes, dtype='uint8')

                # Find the fvalues of each epidermis: assumes adaxial epidermis is at the top of the image
                adaxial_epidermis_value = unique_epidermis_volumes[100, :, 100][(
                    unique_epidermis_volumes[100, :, 100] != 0).argmax()]
                abaxial_epidermis_value = int(np.arange(start=1, stop=3)[
                                              np.arange(start=1, stop=3) != adaxial_epidermis_value])

                # Compute volume
                epidermis_adaxial_volume = epidermis_area[adaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)
                epidermis_abaxial_volume = epidermis_area[abaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)

                # Tichkness return a 2D array, i.e. the thcikness of each column
                epidermis_abaxial_thickness = np.sum(
                    (unique_epidermis_volumes == abaxial_epidermis_value), axis=1) * (px_edge*2)
                epidermis_adaxial_thickness = np.sum(
                    (unique_epidermis_volumes == adaxial_epidermis_value), axis=1) * (px_edge*2)
                del props_of_unique_epidermis
                gc.collect()

                ###################
                ## VEINS
                ###################
                print('### VEINS ###')
                # Get the veins volumes
                unique_vein_volumes = label(raw_pred_stack == vein_value, connectivity=1)
                props_of_unique_veins = regionprops(unique_vein_volumes)

                # io.imshow(unique_vein_volumes[100])

                veins_area = np.zeros(len(props_of_unique_veins))
                veins_label = np.zeros(len(props_of_unique_veins))
                veins_centroid = np.zeros([len(props_of_unique_veins), 3])
                for regions in np.arange(len(props_of_unique_veins)):
                    veins_area[regions] = props_of_unique_veins[regions].area
                    veins_label[regions] = props_of_unique_veins[regions].label
                    veins_centroid[regions] = props_of_unique_veins[regions].centroid

                # Find the largest veins
                ordered_veins = np.argsort(veins_area)
                #veins_area[ordered_veins[-80:]]
                #veins_area[ordered_veins[:1000]]
                #veins_centroid[ordered_veins[-4:]]

                #print(np.sum(veins_area <= 1000))

                # I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
                # the noise left by the segmentation method and kept only the largest veins.
                # This should be adjusted depending on the species/images/maginification.
                large_veins_ids = veins_label[veins_area > (100000/8)]

                largest_veins = np.in1d(unique_vein_volumes, large_veins_ids).reshape(raw_pred_stack.shape)

                del unique_vein_volumes

                # Get the values again
                vein_volume = np.sum(largest_veins) * (px_edge * (px_edge*2)**2)
                del props_of_unique_veins
                gc.collect()

                #Check if it's correct
                #io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_veins.tif',
                #          img_as_ubyte(largest_veins))
                # io.imshow(largest_veins[100])


                ###################
                ## BUNDLE SHEATHS
                ###################
                print('### BUNDLE SHEATHS ###')
                if bs_value > 0:
                    # Get the bs volumes
                    unique_bs_volumes = label(raw_pred_stack == bs_value, connectivity=1)
                    props_of_unique_bs = regionprops(unique_bs_volumes)

                    # io.imshow(unique_bs_volumes[100])

                    bs_area = np.zeros(len(props_of_unique_bs))
                    bs_label = np.zeros(len(props_of_unique_bs))
                    bs_centroid = np.zeros([len(props_of_unique_bs), 3])
                    for regions in np.arange(len(props_of_unique_bs)):
                        bs_area[regions] = props_of_unique_bs[regions].area
                        bs_label[regions] = props_of_unique_bs[regions].label
                        bs_centroid[regions] = props_of_unique_bs[regions].centroid

                    # Find the largest bs
                    ordered_bs = np.argsort(bs_area)
                    #bs_area[ordered_bs[-80:]]
                    #bs_area[ordered_bs[:1000]]
                    #bs_centroid[ordered_bs[-4:]]

                    #print(np.sum(bs_area <= 1000))

                    # I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
                    # the noise left by the segmentation method and kept only the largest bs.
                    # This should be adjusted depending on the species/images/maginification.
                    large_bs_ids = bs_label[bs_area > (100000/8)]

                    largest_bs = np.in1d(unique_bs_volumes, large_bs_ids).reshape(raw_pred_stack.shape)

                    del unique_bs_volumes

                    # Get the values again
                    bs_volume = np.sum(largest_bs) * (px_edge * (px_edge*2)**2)
                    del props_of_unique_bs
                    gc.collect()

                    #Check if it's correct
                    #io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_bs.tif',
                    #          img_as_ubyte(largest_bs))
                    # io.imshow(largest_bs[100])
                else:
                    print('bundle sheath not labelled -- skipped')

                ###################
                ## AIRSPACE
                ###################

                #########################################
                ## CREATE THE FULLSIZE SEGMENTED STACK ##
                #########################################

                # My segmenteation procedure used a reduced size stack, since my original
                # images are too big to be handled. I do want to use my original images for
                # their quality and details, so I use the binary image and add on top of it
                # the background, epidermis, and veins that have been segmented. That way, I
                # keep the detail I want at the airspace-cell interface, while still having a
                # good background, epidermis, and vein segmentation to remove the tissues that
                # are not need for some traits.

                if(reuse_raw_binary == 'True'):
                    ##############################
                    ## LOADING THE BINARY STACK ##
                    ## IN ORIGINAL SIZE         ##
                    ##############################
                    print('')
                    print('### LOADING ORIGINAL SIZED BINARY STACK ###')
                    binary_stack = img_as_bool(io.imread(filepath + binary_filename))
                    binary_stack, to_trim = Trim_Individual_Stack(binary_stack, to_resize)
                    
                    if len(binary_stack.shape) == 4:
                        binary_stack = binary_stack[:,:,:,0]

                    # Trim at the edges -- The ML does a bad job there
                    if trim_slices == 0:
                        if trim_column_L == 0:
                            if trim_column_R == 0:
                                binary_stack = binary_stack
                    else:
                        if trim_column_L == 0:
                            if trim_column_R == 0:
                                binary_stack = binary_stack[trim_slices:-trim_slices, :, :]
                        else:
                            binary_stack = binary_stack[trim_slices:-trim_slices, :, (trim_column_L*2):(-trim_column_R*2)]

                    # Check and trim the binary stack if necessary
                    # This is to match the dimensions between all images
                    # Basically, it trims odds numbered dimension so to be able to divide/multiply them by 2.


                else:
                    print('### USING PREDICTIONS INSTEAD OF ORIGINAL BINARY STACK ###')

                if(to_resize == 'False'):
                    if(reuse_raw_binary == 'True'):
                        new_shape = binary_stack.shape
                    new_shape = raw_pred_stack.shape
                else:
                    new_shape = tuple([raw_pred_stack.shape[0], raw_pred_stack.shape[1] * int(to_resize), raw_pred_stack.shape[2] * int(to_resize)])

                # This cell creates an empty array filled with the backgroud color (177), then
                # adds all of the leaf to it. Looping over each slice (this is more memory
                # efficient than working on the whole stack), it takes the ML segmented image,
                # resize the slice, and adds it to the empty array.
                bg_value_new = 177
                vein_value_new = 147
                ias_value_new = 255
                bs_value_new = 102
                mesophyll_value_new = 0

                print('### CREATING THE POST-PROCESSED SEGMENTED STACK ###')

                # Assign an array filled with the background value 177.
                large_segmented_stack = np.full(shape=new_shape, fill_value=bg_value_new, dtype='uint8')
                for idx in np.arange(large_segmented_stack.shape[0]):
                    # Creates a boolean 2D array of the veins (from the largest veins id'ed earlier)
                    temp_veins = img_as_bool(transform.resize(largest_veins[idx],
                                                              [new_shape[1], new_shape[2]],
                                                              anti_aliasing=False, order=0))
                    # Creates a boolean 2D array of the bundle sheath (from the largest veins id'ed earlier)
                    if bs_value > 0:
                        temp_bs = img_as_bool(transform.resize(largest_bs[idx],
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                    # Creates a 2D array with the epidermis being assinged values 30 or 60
                    temp_epid = transform.resize(unique_epidermis_volumes[idx],
                                                 [new_shape[1], new_shape[2]],
                                                 anti_aliasing=False, preserve_range=True, order=0) * 30
                    # Creates a 2D mask of only the leaf to remove the backgroud from the
                    # original sized binary image.
                    leaf_mask = img_as_bool(transform.resize(raw_pred_stack[idx] != bg_value,
                                                             [new_shape[1], new_shape[2]],
                                                             anti_aliasing=False, order=0))
                    # binary_stack is a boolean, so you need to multiply it.
                    if(reuse_raw_binary == 'True'):
                        large_segmented_stack[idx][leaf_mask] = binary_stack[idx][leaf_mask] * ias_value_new
                    else:
                        temp_cells = img_as_bool(transform.resize(raw_pred_stack[idx] == mesophyll_value,
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                        temp_air = img_as_bool(transform.resize(raw_pred_stack[idx] == ias_value,
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                        large_segmented_stack[idx][leaf_mask] = temp_cells[leaf_mask] * mesophyll_value_new
                        large_segmented_stack[idx][leaf_mask] = temp_air[leaf_mask] * ias_value_new

                    large_segmented_stack[idx][temp_veins] = vein_value_new
                    if bs_value > 0:
                        large_segmented_stack[idx][temp_bs] = bs_value_new
                    large_segmented_stack[idx][temp_epid != 0] = temp_epid[temp_epid != 0]

                # io.imshow(large_segmented_stack[100])
                print("")
                print('### Validate the values in the stack ###')
                print((np.unique(large_segmented_stack[100])))

                # Special tiff saving option for ImageJ compatibility when files larger than
                # 2 Gb. It's like it doesn't recognize something if you don't turn this option
                # on for large files and then ImageJ or FIJI fail to load the large stack
                # (happens on my linux machine installed with openSUSE Tumbleweed).
                if large_segmented_stack.nbytes >= 2e9:
                    imgj_bool = True
                else:
                    imgj_bool = False

                # Save the image
                print("")
                print('### Saving post-processed segmented stack ###')
                io.imsave(base_folder_name + sample_name + '/' + sample_name
                          + 'SEGMENTED.tif', large_segmented_stack, imagej=imgj_bool)


            ################################################
            ## COMPUTE TRAITS ON THE ORIGINAL SIZED STACK ##
            ################################################
            print('')
            print('### COMPUTE TRAITS ###')
            print('')
            # Redefine the values for the different tissues as used in the segmented image.
            # The epidermis will be defined later.
            bg_value = 177
            spongy_value = 0
            palisade_value = 0
            if spongy_value == palisade_value:
                mesophyll_value = spongy_value
            else:
                mesophyll_value = [spongy_value, palisade_value]
            ias_value = 255
            vein_value = 147
            bs_value = 102

            # Find the values of each epidermis: assumes adaxial epidermis is at the top of the image
            # Find the values of each epidermis: assumes adaxial epidermis is at the top of the image
            epid_vals = [30,60]
            epid_bool = [i in epid_vals for i in large_segmented_stack[200,:,200]]
            epid_indx = [i for i, x in enumerate(epid_bool) if x]

            adaxial_epidermis_value = large_segmented_stack[200,epid_indx[0],200]
            # adaxial_epidermis_value = large_segmented_stack[100, :, 100][(
            #     large_segmented_stack[100, :, 100] != bg_value).argmax()]

            if adaxial_epidermis_value == 30:
                abaxial_epidermis_value = 60
            else:
                if adaxial_epidermis_value == 60:
                    abaxial_epidermis_value = 30

            #Measure the different volumes

            # Somehow those two lines can give negative values in some cases
            # I expect it's because of bad labelling of the background.
            # leaf_volume = np.sum(large_segmented_stack != bg_value) * vx_volume
            # mesophyll_volume = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack
            #                                                                  != adaxial_epidermis_value) & (large_segmented_stack != abaxial_epidermis_value)) * vx_volume
            cell_volume = np.sum(large_segmented_stack == mesophyll_value) * vx_volume
            air_volume = np.sum(large_segmented_stack == ias_value) * vx_volume
            epidermis_abaxial_volume = np.sum(large_segmented_stack == abaxial_epidermis_value) * vx_volume
            epidermis_adaxial_volume = np.sum(large_segmented_stack == adaxial_epidermis_value) * vx_volume
            vein_volume = np.sum(large_segmented_stack == vein_value) * vx_volume
            bundle_sheath_volume = np.sum(large_segmented_stack == bs_value) * vx_volume
            mesophyll_volume = cell_volume + air_volume
            leaf_volume = mesophyll_volume + epidermis_abaxial_volume + epidermis_adaxial_volume + vein_volume + bundle_sheath_volume

            #Measure the thickness of the leaf, the epidermis, and the mesophyll
            leaf_thickness = np.sum(np.array(large_segmented_stack
                                             != bg_value, dtype='bool'), axis=1) * px_edge
            mesophyll_thickness = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != adaxial_epidermis_value) & (
                large_segmented_stack != abaxial_epidermis_value), axis=1) * px_edge
            epidermis_abaxial_thickness = np.sum(
                large_segmented_stack == abaxial_epidermis_value, axis=1) * px_edge
            epidermis_adaxial_thickness = np.sum(
                large_segmented_stack == adaxial_epidermis_value, axis=1) * px_edge

            print('Leaf thickness')
            print((np.median(leaf_thickness), leaf_thickness.mean(), leaf_thickness.std()))
            print('Mesophyll thickness')
            print((np.median(mesophyll_thickness), mesophyll_thickness.mean(), mesophyll_thickness.std()))
            print('Epidermis thickness (Adaxial)')
            print((np.median(epidermis_adaxial_thickness),
                   epidermis_adaxial_thickness.mean(), epidermis_adaxial_thickness.std()))
            print('Epidermis thickness (Abaxial)')
            print((np.median(epidermis_abaxial_thickness),
                   epidermis_abaxial_thickness.mean(), epidermis_abaxial_thickness.std()))


            # Leaf area
            # I was lazy here as I assume the leaf is parallel to the border of the image.
            leaf_area = large_segmented_stack.shape[0] * large_segmented_stack.shape[2] * (px_edge**2)

            #Caluculate Surface Area (adapted from Matt Jenkins' code)
            # This can take quite a lot of RAM!!!
            # This gives very similar results to BoneJ. BoneJ uses the Lorensen algorithm,
            # which is available as use_classic=True in te marching_cubes_lewiner() function.
            # However, the help page for this function warns that the Lorensen algorithm
            # might result in topologically incorrect results, and as such the Lewiner
            # algorithm is better (and faster too). So it is probably a better approach.
            # , spacing=(px_edge,px_edge,px_edge))
            print("")
            print('### Compute surface area of IAS ###')
            print('### This may take a while and freeze your computer ###')

            ias_vert_faces = marching_cubes_lewiner(
                large_segmented_stack == ias_value, 0, allow_degenerate=False, step_size=2)
            ias_SA = mesh_surface_area(ias_vert_faces[0], ias_vert_faces[1])
            true_ias_SA = ias_SA * (px_edge**2)
            print(('IAS surface area: '+str(true_ias_SA)+' '+units+'**2'))
            if units == 'um':
                print(('or '+str(float(true_ias_SA/1000000))+' mm**2'))
            # end Matt's code adaptation


            try:
                bs_volume
            except NameError:
                bs_volume = 0

            print(('Sm: '+str(true_ias_SA/leaf_area)))
            print(('SAmes/Vmes: '+str(true_ias_SA/mesophyll_volume)))

            # NOTE ON SA CODE ABOVE
            # The same procedure as for epidermises and veins could be done, i.e. using the
            # label() function to identify all of the un-connected airspace volumes and
            # compute the surface area on each of them. That way we can get the surface
            # area of the largest airspace and the connectivity term presented in Earles
            # et al. (2018) (Beyond porosity - Bromeliad paper).

            # This is done in the code below.

            ## Label all of the ias regions
            #unique_ias_volumes = label(large_segmented_stack == ias_value, connectivity=1)
            #props_of_unique_ias = regionprops(unique_ias_volumes)
            #
            ## Find the size and properties of the epidermis regions
            #ias_area = np.zeros(len(props_of_unique_ias))
            #ias_label = np.zeros(len(props_of_unique_ias))
            #ias_centroid = np.zeros([len(props_of_unique_ias),3])
            #ias_SA = np.zeros([len(props_of_unique_ias),3])
            #for regions in np.arange(len(props_of_unique_ias)):
            #    ias_area[regions] = props_of_unique_ias[regions].area
            #    ias_label[regions] = props_of_unique_ias[regions].label
            #    ias_centroid[regions] = props_of_unique_ias[regions].centroid
            #
            ## Find the two largest ias
            #ordered_ias = np.argsort(ias_area)
            #print('Check for the largest values and adjust use the necessary value to compute the largest connected IAS.')
            #print(ias_area[ordered_ias[-4:]])
            #print(ias_label[ordered_ias[-4:]])
            #
            #largests_ias = ias_label[ordered_ias[-4:]]
            #ias_SA = np.zeros(len(largests_ias))
            #for i in np.arange(len(ias_SA)):
            #    ias_vert_faces = marching_cubes_lewiner(unique_ias_volumes == largests_ias[i], 0, spacing=(px_edge,px_edge,px_edge))
            #    ias_SA[i] = mesh_surface_area(ias_vert_faces[0],ias_vert_faces[1])
            #    print(ias_SA[i])
            #
            ## Adjust depending on if you have IAS that are large but not connected such as
            ##  when bundle sheath extensions are present.
            #largest_connected_ias_SA = sum(ias_SA[-1:])
            #largest_connected_ias_volume = sum(ias_area[ordered_ias[-1:]]) * vx_volume


            # Write the data into a data frame
            data_out = {'LeafArea': leaf_area,
                        'LeafThickness': leaf_thickness.mean(),
                        'LeafThickness_SD': leaf_thickness.std(),
                        'MesophyllThickness': mesophyll_thickness.mean(),
                        'MesophyllThickness_SD': mesophyll_thickness.std(),
                        'ADEpidermisThickness': epidermis_adaxial_thickness.mean(),
                        'ADEpidermisThickness_SD': epidermis_adaxial_thickness.std(),
                        'ABEpidermisThickness': epidermis_abaxial_thickness.mean(),
                        'ABEpidermisThickness_SD': epidermis_abaxial_thickness.std(),
                        'LeafVolume': leaf_volume,
                        'MesophyllVolume': mesophyll_volume,
                        'ADEpidermisVolume': epidermis_adaxial_volume,
                        'ABEpidermisVolume': epidermis_abaxial_volume,
                        'VeinVolume': vein_volume,
                        'BSVolume': bs_volume,
                        'VeinBSVolume': vein_volume+bs_volume,
                        'CellVolume': cell_volume,
                        'IASVolume': air_volume,
                        'IASSurfaceArea': true_ias_SA,
                        'PixelSize': px_edge,
                        'Units': units,
                        '_SLICEStrimmed':trim_slices,
                        '_X_trimmed_left':trim_column_L*to_resize,
                        '_X_trimmed_right':trim_column_L*to_resize}
                        #            'IASLargestConnectedVolume':largest_connected_ias_volume,
                        #            'IASLargestConnectedSA':largest_connected_ias_SA

            results_out = DataFrame(data_out, index={sample_name})
            # Save the data to a CSV
            print('### Saving results to text file ###')
            results_out.to_csv(base_folder_name + sample_name + '/' + sample_name
                               + 'RESULTS.txt', sep='\t', encoding='utf-8')
            print('Data saved')
            for key, value in data_out.items():
                if isinstance(value, str):
                    print(str(key) + ' : ' + value)
                else:
                    print(str(key) + ' : ' + str(round(value, 3)))
            print('')
            print('Done with ' + sample_name)
            print('')
        #
            j += 1
    else:
        # check for definitions of all required parameters before proceeding
        try: sample_name, px_edge, to_resize, units, reuse_raw_binary, trim_slices, color_values, base_folder_name
        except NameError: req_not_def = 1
        if reuse_raw_binary=='True':
            try: binary_postfix
            except NameError: req_not_def = 1

        if req_not_def==0:
            print('\nSingle scan mode...\n')
            #
            # Pixel dimmension
            vx_volume = px_edge**3
        # TESTING
                # print(vx_volume)
                # print(color_values)
        # TESTING

            # Define the different tissue values
            epid_value, bg_value, mesophyll_value, ias_value, vein_value, bs_value = [int(x) for x in color_values.split(',')]

        # TESTING
                # print(epid_value, bg_value, mesophyll_value)
        # TESTING

            # Load segmented image
            # Set directory of functions in order to import MLmicroCTfunctions
            # path_to_script = '/'.join(path.split('/')[:-1]) + '/'
            # os.chdir(path_to_script)

            sample_path_split = sample_name.split('/')
        # TESTING
                # print(sample_path_split)
        # TESTING

            # If input path to sample is of length 1, i.e. only the sample name,
            # create the folder names based on default file naming.
            if len(sample_path_split) == 1:
                folder_name = '/MLresults/'
                raw_ML_prediction_name = sample_name + 'fullstack_prediction.tif'
            else:
                sample_name = sample_path_split[-3]
                folder_name = sample_path_split[-2] + '/'
                raw_ML_prediction_name = sample_path_split[-1]
            filepath = base_folder_name + sample_name + '/'
            if reuse_raw_binary == 'True':
                binary_filename = sample_name + binary_postfix
            # raw_ML_prediction_name = sample_name + 'fullstack_prediction.tif'

            print('')
            print('#################')
            print('# STARTING WITH #')
            print('#################')
            print(' ' + sample_name)

        #
            # Check if the file has already been processed -- Just in case!
            if os.path.isfile(filepath + sample_name + 'RESULTS.txt'):
                print('')
                print('This file has already been processed!')
                print('')
                assert False

            if os.path.isfile(base_folder_name + sample_name + '/' + sample_name + 'SEGMENTED.tif'):
                print('###LOADING POST-PROCESSED SEGMENTED STACK###')
                large_segmented_stack = io.imread(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif')
            else:
                # Load the ML segmented stack
                raw_pred_stack = io.imread(filepath + folder_name + raw_ML_prediction_name)
                uniq100th = np.unique(raw_pred_stack[100])

                if np.any(uniq100th < 0):
                    raw_pred_stack = np.where(raw_pred_stack < 0, raw_pred_stack + 256, raw_pred_stack)
                    print(np.unique(raw_pred_stack[100]))
                else:
                    print(uniq100th)

                # Trim at the edges -- The ML does a bad job there
                if trim_slices == 0:
                    if trim_column_L == 0:
                        if trim_column_R == 0:
                            raw_pred_stack = raw_pred_stack
                else:
                    if trim_column_L == 0:
                        if trim_column_R == 0:
                            raw_pred_stack = raw_pred_stack = raw_pred_stack[trim_slices:-trim_slices, :, :]
                    else:
                        raw_pred_stack = raw_pred_stack[trim_slices:-trim_slices, :, trim_column_L:-trim_column_R]
    #
    #
                ###################
                # EPIDERMIS
                ###################
                print('')
                print('### EPIDERMIS ###')
                print('')

                # Label all of the epidermis regions
                print('labeling')
                unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
                props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

                # io.imshow(unique_epidermis_volumes[100])

                # Find the size and properties of the epidermis regions
                print('finding regions')
                epidermis_area = np.zeros(len(props_of_unique_epidermis))
                epidermis_label = np.zeros(len(props_of_unique_epidermis))
                epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                print('looping over regions')
                for regions in np.arange(len(props_of_unique_epidermis)):
                    epidermis_area[regions] = props_of_unique_epidermis[regions].area
                    epidermis_label[regions] = props_of_unique_epidermis[regions].label
                    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                # Find the two largest epidermis
                ordered_epidermis = np.argsort(epidermis_area)
                print('The two largest values below should be in the same order of magnitude')
                print((epidermis_area[ordered_epidermis[-4:]]))

                if epidermis_area[ordered_epidermis[-1]] > (10*epidermis_area[ordered_epidermis[-2]]):
                    print('ERROR: Both epidermis might be connected!')
                    print('Trying to remove the dangling epidermis pixels')
                    no_dangling = delete_dangling_epidermis((unique_epidermis_volumes == ordered_epidermis[-1]+1), True, False)

                    # Label all of the epidermis regions
                    unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
                    props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

                    # io.imshow(unique_epidermis_volumes[100])

                    # Find the size and properties of the epidermis regions
                    epidermis_area = np.zeros(len(props_of_unique_epidermis))
                    epidermis_label = np.zeros(len(props_of_unique_epidermis))
                    epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                    for regions in np.arange(len(props_of_unique_epidermis)):
                        epidermis_area[regions] = props_of_unique_epidermis[regions].area
                        epidermis_label[regions] = props_of_unique_epidermis[regions].label
                        epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                    # Find the two largest epidermis
                    ordered_epidermis = np.argsort(epidermis_area)
                    print('The two largest values below should be in the same order of magnitude')
                    print((epidermis_area[ordered_epidermis[-4:]]))

                    if epidermis_area[ordered_epidermis[-1]] > (10*epidermis_area[ordered_epidermis[-2]]):
                        print('#########################################')
                        print('#########################################')
                        print('ERROR: Both epidermis are still connected!')
                        print('' + sample_name)
                        print('#########################################')
                        print('#########################################')
                        assert False

                print("")
                print('The center of the epidermis should be more or less the same on the')
                print('1st and 3rd columns for the two largest values.')
                print((epidermis_centroid[ordered_epidermis[-2:]]))
                print("")

                two_largest_epidermis = (unique_epidermis_volumes
                                         == ordered_epidermis[-1]+1) | (unique_epidermis_volumes == ordered_epidermis[-2]+1)

                #Check if it's correct
                #io.imsave(filepath + folder_name + 'test_epidermis.tif',
                #          img_as_ubyte(two_largest_epidermis))
                # io.imshow(two_largest_epidermis[100])


                # Get the values again: makes it cleaner
                unique_epidermis_volumes = label(two_largest_epidermis, connectivity=1)
                props_of_unique_epidermis = regionprops(unique_epidermis_volumes)
                epidermis_area = np.zeros(len(props_of_unique_epidermis))
                epidermis_label = np.zeros(len(props_of_unique_epidermis))
                epidermis_centroid = np.zeros([len(props_of_unique_epidermis), 3])
                for regions in np.arange(len(props_of_unique_epidermis)):
                    epidermis_area[regions] = props_of_unique_epidermis[regions].area
                    epidermis_label[regions] = props_of_unique_epidermis[regions].label
                    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

                ## io.imshow(unique_epidermis_volumes[100])

                # Transform the array to 8-bit: no need for the extra precision as there are only 3 values
                unique_epidermis_volumes = np.array(unique_epidermis_volumes, dtype='uint8')

                print('>>>> SAVING EPIDERMIS STACK')
                io.imsave(base_folder_name + sample_name + '/' + sample_name
                          + 'TWO_EPIDERMIS.tif', unique_epidermis_volumes, imagej=imgj_bool)

                # Find the fvalues of each epidermis: assumes adaxial epidermis is at the top of the image
                adaxial_epidermis_value = unique_epidermis_volumes[100, :, 100][(
                    unique_epidermis_volumes[100, :, 100] != 0).argmax()]
                abaxial_epidermis_value = int(np.arange(start=1, stop=3)[
                                              np.arange(start=1, stop=3) != adaxial_epidermis_value])

                # Compute volume
                epidermis_adaxial_volume = epidermis_area[adaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)
                epidermis_abaxial_volume = epidermis_area[abaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)

                # Tichkness return a 2D array, i.e. the thcikness of each column
                epidermis_abaxial_thickness = np.sum(
                    (unique_epidermis_volumes == abaxial_epidermis_value), axis=1) * (px_edge*2)
                epidermis_adaxial_thickness = np.sum(
                    (unique_epidermis_volumes == adaxial_epidermis_value), axis=1) * (px_edge*2)
                del props_of_unique_epidermis
                gc.collect()

                ###################
                ## VEINS
                ###################
                print('### VEINS ###')
                # Get the veins volumes
                unique_vein_volumes = label(raw_pred_stack == vein_value, connectivity=1)
                props_of_unique_veins = regionprops(unique_vein_volumes)

                # io.imshow(unique_vein_volumes[100])

                veins_area = np.zeros(len(props_of_unique_veins))
                veins_label = np.zeros(len(props_of_unique_veins))
                veins_centroid = np.zeros([len(props_of_unique_veins), 3])
                for regions in np.arange(len(props_of_unique_veins)):
                    veins_area[regions] = props_of_unique_veins[regions].area
                    veins_label[regions] = props_of_unique_veins[regions].label
                    veins_centroid[regions] = props_of_unique_veins[regions].centroid

                # Find the largest veins
                ordered_veins = np.argsort(veins_area)
                #veins_area[ordered_veins[-80:]]
                #veins_area[ordered_veins[:1000]]
                #veins_centroid[ordered_veins[-4:]]

                #print(np.sum(veins_area <= 1000))

                # I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
                # the noise left by the segmentation method and kept only the largest veins.
                # This should be adjusted depending on the species/images/maginification.
                large_veins_ids = veins_label[veins_area > (100000/8)]

                largest_veins = np.in1d(unique_vein_volumes, large_veins_ids).reshape(raw_pred_stack.shape)

                del unique_vein_volumes

                # Get the values again
                vein_volume = np.sum(largest_veins) * (px_edge * (px_edge*2)**2)
                del props_of_unique_veins
                gc.collect()

                #Check if it's correct
                #io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_veins.tif',
                #          img_as_ubyte(largest_veins))
                # io.imshow(largest_veins[100])


                ###################
                ## BUNDLE SHEATHS
                ###################
                print('### BUNDLE SHEATHS ###')
                if bs_value > 0:
                    # Get the bs volumes
                    unique_bs_volumes = label(raw_pred_stack == bs_value, connectivity=1)
                    props_of_unique_bs = regionprops(unique_bs_volumes)

                    # io.imshow(unique_bs_volumes[100])

                    bs_area = np.zeros(len(props_of_unique_bs))
                    bs_label = np.zeros(len(props_of_unique_bs))
                    bs_centroid = np.zeros([len(props_of_unique_bs), 3])
                    for regions in np.arange(len(props_of_unique_bs)):
                        bs_area[regions] = props_of_unique_bs[regions].area
                        bs_label[regions] = props_of_unique_bs[regions].label
                        bs_centroid[regions] = props_of_unique_bs[regions].centroid

                    # Find the largest bs
                    ordered_bs = np.argsort(bs_area)
                    #bs_area[ordered_bs[-80:]]
                    #bs_area[ordered_bs[:1000]]
                    #bs_centroid[ordered_bs[-4:]]

                    #print(np.sum(bs_area <= 1000))

                    # I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
                    # the noise left by the segmentation method and kept only the largest bs.
                    # This should be adjusted depending on the species/images/maginification.
                    large_bs_ids = bs_label[bs_area > (100000/8)]

                    largest_bs = np.in1d(unique_bs_volumes, large_bs_ids).reshape(raw_pred_stack.shape)

                    del unique_bs_volumes

                    # Get the values again
                    bs_volume = np.sum(largest_bs) * (px_edge * (px_edge*2)**2)
                    del props_of_unique_bs
                    gc.collect()

                    #Check if it's correct
                    #io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_bs.tif',
                    #          img_as_ubyte(largest_bs))
                    # io.imshow(largest_bs[100])
                else:
                    print('bundle sheath not labelled -- skipped')

                ###################
                ## AIRSPACE
                ###################

                #########################################
                ## CREATE THE FULLSIZE SEGMENTED STACK ##
                #########################################

                # My segmenteation procedure used a reduced size stack, since my original
                # images are too big to be handled. I do want to use my original images for
                # their quality and details, so I use the binary image and add on top of it
                # the background, epidermis, and veins that have been segmented. That way, I
                # keep the detail I want at the airspace-cell interface, while still having a
                # good background, epidermis, and vein segmentation to remove the tissues that
                # are not need for some traits.

                if(reuse_raw_binary == 'True'):
                    ##############################
                    ## LOADING THE BINARY STACK ##
                    ## IN ORIGINAL SIZE         ##
                    ##############################
                    print('')
                    print('### LOADING ORIGINAL SIZED BINARY STACK ###')
                    binary_stack = img_as_bool(io.imread(filepath + binary_filename))
                    binary_stack, to_trim = Trim_Individual_Stack(binary_stack, int(to_resize))
                    
                    if len(binary_stack.shape) == 4:
                        binary_stack = binary_stack[:,:,:,0]

                    # Trim at the edges -- The ML does a bad job there
                    if trim_slices == 0:
                        if trim_column_L == 0:
                            if trim_column_R == 0:
                                binary_stack = binary_stack
                    else:
                        if trim_column_L == 0:
                            if trim_column_R == 0:
                                binary_stack = binary_stack[trim_slices:-trim_slices, :, :]
                        else:
                            binary_stack = binary_stack[trim_slices:-trim_slices, :, (trim_column_L*2):(-trim_column_R*2)]

                    # Check and trim the binary stack if necessary
                    # This is to match the dimensions between all images
                    # Basically, it trims odds numbered dimension so to be able to divide/multiply them by 2.


                else:
                    print('### USING PREDICTIONS INSTEAD OF ORIGINAL BINARY STACK ###')

                if(to_resize == 'False'):
                    if(reuse_raw_binary == 'True'):
                        new_shape = binary_stack.shape
                    new_shape = raw_pred_stack.shape
                else:
                    new_shape = tuple([raw_pred_stack.shape[0], raw_pred_stack.shape[1] * int(to_resize), raw_pred_stack.shape[2] * int(to_resize)])

                # This cell creates an empty array filled with the backgroud color (177), then
                # adds all of the leaf to it. Looping over each slice (this is more memory
                # efficient than working on the whole stack), it takes the ML segmented image,
                # resize the slice, and adds it to the empty array.
                bg_value_new = 177
                vein_value_new = 147
                ias_value_new = 255
                bs_value_new = 102
                mesophyll_value_new = 0

                print('### CREATING THE POST-PROCESSED SEGMENTED STACK ###')

                # Assign an array filled with the background value 177.
                large_segmented_stack = np.full(shape=new_shape, fill_value=bg_value_new, dtype='uint8')
                for idx in np.arange(large_segmented_stack.shape[0]):
                    # Creates a boolean 2D array of the veins (from the largest veins id'ed earlier)
                    temp_veins = img_as_bool(transform.resize(largest_veins[idx],
                                                              [new_shape[1], new_shape[2]],
                                                              anti_aliasing=False, order=0))
                    # Creates a boolean 2D array of the bundle sheath (from the largest veins id'ed earlier)
                    if bs_value > 0:
                        temp_bs = img_as_bool(transform.resize(largest_bs[idx],
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                    # Creates a 2D array with the epidermis being assinged values 30 or 60
                    temp_epid = transform.resize(unique_epidermis_volumes[idx],
                                                 [new_shape[1], new_shape[2]],
                                                 anti_aliasing=False, preserve_range=True, order=0) * 30
                    # Creates a 2D mask of only the leaf to remove the backgroud from the
                    # original sized binary image.
                    leaf_mask = img_as_bool(transform.resize(raw_pred_stack[idx] != bg_value,
                                                             [new_shape[1], new_shape[2]],
                                                             anti_aliasing=False, order=0))
                    # binary_stack is a boolean, so you need to multiply it.
                    if(reuse_raw_binary == 'True'):
                        large_segmented_stack[idx][leaf_mask] = binary_stack[idx][leaf_mask] * ias_value_new
                    else:
                        temp_cells = img_as_bool(transform.resize(raw_pred_stack[idx] == mesophyll_value,
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                        temp_air = img_as_bool(transform.resize(raw_pred_stack[idx] == ias_value,
                                                               [new_shape[1], new_shape[2]],
                                                               anti_aliasing=False, order=0))
                        large_segmented_stack[idx][leaf_mask] = temp_cells[leaf_mask] * mesophyll_value_new
                        large_segmented_stack[idx][leaf_mask] = temp_air[leaf_mask] * ias_value_new

                    large_segmented_stack[idx][temp_veins] = vein_value_new
                    if bs_value > 0:
                        large_segmented_stack[idx][temp_bs] = bs_value_new
                    large_segmented_stack[idx][temp_epid != 0] = temp_epid[temp_epid != 0]

                # io.imshow(large_segmented_stack[100])
                print("")
                print('### Validate the values in the stack ###')
                print((np.unique(large_segmented_stack[100])))

                # Special tiff saving option for ImageJ compatibility when files larger than
                # 2 Gb. It's like it doesn't recognize something if you don't turn this option
                # on for large files and then ImageJ or FIJI fail to load the large stack
                # (happens on my linux machine installed with openSUSE Tumbleweed).
                if large_segmented_stack.nbytes >= 2e9:
                    imgj_bool = True
                else:
                    imgj_bool = False

                # Save the image
                print("")
                print('### Saving post-processed segmented stack ###')
                io.imsave(base_folder_name + sample_name + '/' + sample_name
                          + 'SEGMENTED.tif', large_segmented_stack, imagej=imgj_bool)


            ################################################
            ## COMPUTE TRAITS ON THE ORIGINAL SIZED STACK ##
            ################################################
            print('')
            print('### COMPUTE TRAITS ###')
            print('')
            # Redefine the values for the different tissues as used in the segmented image.
            # The epidermis will be defined later.
            bg_value = 177
            spongy_value = 0
            palisade_value = 0
            if spongy_value == palisade_value:
                mesophyll_value = spongy_value
            else:
                mesophyll_value = [spongy_value, palisade_value]
            ias_value = 255
            vein_value = 147
            bs_value = 102

            # Find the values of each epidermis: assumes adaxial epidermis is at the top of the image
            # Find the values of each epidermis: assumes adaxial epidermis is at the top of the image
            epid_vals = [30,60]
            epid_bool = [i in epid_vals for i in large_segmented_stack[200,:,200]]
            epid_indx = [i for i, x in enumerate(epid_bool) if x]

            adaxial_epidermis_value = large_segmented_stack[200,epid_indx[0],200]
            # adaxial_epidermis_value = large_segmented_stack[100, :, 100][(
            #     large_segmented_stack[100, :, 100] != bg_value).argmax()]

            if adaxial_epidermis_value == 30:
                abaxial_epidermis_value = 60
            else:
                if adaxial_epidermis_value == 60:
                    abaxial_epidermis_value = 30

            #Measure the different volumes

            # Somehow those two lines can give negative values in some cases
            # I expect it's because of bad labelling of the background.
            # leaf_volume = np.sum(large_segmented_stack != bg_value) * vx_volume
            # mesophyll_volume = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack
            #                                                                  != adaxial_epidermis_value) & (large_segmented_stack != abaxial_epidermis_value)) * vx_volume
            cell_volume = np.sum(large_segmented_stack == mesophyll_value) * vx_volume
            air_volume = np.sum(large_segmented_stack == ias_value) * vx_volume
            epidermis_abaxial_volume = np.sum(large_segmented_stack == abaxial_epidermis_value) * vx_volume
            epidermis_adaxial_volume = np.sum(large_segmented_stack == adaxial_epidermis_value) * vx_volume
            vein_volume = np.sum(large_segmented_stack == vein_value) * vx_volume
            bundle_sheath_volume = np.sum(large_segmented_stack == bs_value) * vx_volume
            mesophyll_volume = cell_volume + air_volume
            leaf_volume = mesophyll_volume + epidermis_abaxial_volume + epidermis_adaxial_volume + vein_volume + bundle_sheath_volume

            #Measure the thickness of the leaf, the epidermis, and the mesophyll
            leaf_thickness = np.sum(np.array(large_segmented_stack
                                             != bg_value, dtype='bool'), axis=1) * px_edge
            mesophyll_thickness = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != adaxial_epidermis_value) & (
                large_segmented_stack != abaxial_epidermis_value), axis=1) * px_edge
            epidermis_abaxial_thickness = np.sum(
                large_segmented_stack == abaxial_epidermis_value, axis=1) * px_edge
            epidermis_adaxial_thickness = np.sum(
                large_segmented_stack == adaxial_epidermis_value, axis=1) * px_edge

            print('Leaf thickness')
            print((np.median(leaf_thickness), leaf_thickness.mean(), leaf_thickness.std()))
            print('Mesophyll thickness')
            print((np.median(mesophyll_thickness), mesophyll_thickness.mean(), mesophyll_thickness.std()))
            print('Epidermis thickness (Adaxial)')
            print((np.median(epidermis_adaxial_thickness),
                   epidermis_adaxial_thickness.mean(), epidermis_adaxial_thickness.std()))
            print('Epidermis thickness (Abaxial)')
            print((np.median(epidermis_abaxial_thickness),
                   epidermis_abaxial_thickness.mean(), epidermis_abaxial_thickness.std()))


            # Leaf area
            # I was lazy here as I assume the leaf is parallel to the border of the image.
            leaf_area = large_segmented_stack.shape[0] * large_segmented_stack.shape[2] * (px_edge**2)

            #Caluculate Surface Area (adapted from Matt Jenkins' code)
            # This can take quite a lot of RAM!!!
            # This gives very similar results to BoneJ. BoneJ uses the Lorensen algorithm,
            # which is available as use_classic=True in te marching_cubes_lewiner() function.
            # However, the help page for this function warns that the Lorensen algorithm
            # might result in topologically incorrect results, and as such the Lewiner
            # algorithm is better (and faster too). So it is probably a better approach.
            # , spacing=(px_edge,px_edge,px_edge))
            print("")
            print('### Compute surface area of IAS ###')
            print('### This may take a while and freeze your computer ###')

            ias_vert_faces = marching_cubes_lewiner(
                large_segmented_stack == ias_value, 0, allow_degenerate=False, step_size=2)
            ias_SA = mesh_surface_area(ias_vert_faces[0], ias_vert_faces[1])
            true_ias_SA = ias_SA * (px_edge**2)
            print(('IAS surface area: '+str(true_ias_SA)+' '+units+'**2'))
            if units == 'um':
                print(('or '+str(float(true_ias_SA/1000000))+' mm**2'))
            # end Matt's code adaptation


            try:
                bs_volume
            except NameError:
                bs_volume = 0

            print(('Sm: '+str(true_ias_SA/leaf_area)))
            print(('SAmes/Vmes: '+str(true_ias_SA/mesophyll_volume)))

            # NOTE ON SA CODE ABOVE
            # The same procedure as for epidermises and veins could be done, i.e. using the
            # label() function to identify all of the un-connected airspace volumes and
            # compute the surface area on each of them. That way we can get the surface
            # area of the largest airspace and the connectivity term presented in Earles
            # et al. (2018) (Beyond porosity - Bromeliad paper).

            # This is done in the code below.

            ## Label all of the ias regions
            #unique_ias_volumes = label(large_segmented_stack == ias_value, connectivity=1)
            #props_of_unique_ias = regionprops(unique_ias_volumes)
            #
            ## Find the size and properties of the epidermis regions
            #ias_area = np.zeros(len(props_of_unique_ias))
            #ias_label = np.zeros(len(props_of_unique_ias))
            #ias_centroid = np.zeros([len(props_of_unique_ias),3])
            #ias_SA = np.zeros([len(props_of_unique_ias),3])
            #for regions in np.arange(len(props_of_unique_ias)):
            #    ias_area[regions] = props_of_unique_ias[regions].area
            #    ias_label[regions] = props_of_unique_ias[regions].label
            #    ias_centroid[regions] = props_of_unique_ias[regions].centroid
            #
            ## Find the two largest ias
            #ordered_ias = np.argsort(ias_area)
            #print('Check for the largest values and adjust use the necessary value to compute the largest connected IAS.')
            #print(ias_area[ordered_ias[-4:]])
            #print(ias_label[ordered_ias[-4:]])
            #
            #largests_ias = ias_label[ordered_ias[-4:]]
            #ias_SA = np.zeros(len(largests_ias))
            #for i in np.arange(len(ias_SA)):
            #    ias_vert_faces = marching_cubes_lewiner(unique_ias_volumes == largests_ias[i], 0, spacing=(px_edge,px_edge,px_edge))
            #    ias_SA[i] = mesh_surface_area(ias_vert_faces[0],ias_vert_faces[1])
            #    print(ias_SA[i])
            #
            ## Adjust depending on if you have IAS that are large but not connected such as
            ##  when bundle sheath extensions are present.
            #largest_connected_ias_SA = sum(ias_SA[-1:])
            #largest_connected_ias_volume = sum(ias_area[ordered_ias[-1:]]) * vx_volume


            # Write the data into a data frame
            data_out = {'LeafArea': leaf_area,
                        'LeafThickness': leaf_thickness.mean(),
                        'LeafThickness_SD': leaf_thickness.std(),
                        'MesophyllThickness': mesophyll_thickness.mean(),
                        'MesophyllThickness_SD': mesophyll_thickness.std(),
                        'ADEpidermisThickness': epidermis_adaxial_thickness.mean(),
                        'ADEpidermisThickness_SD': epidermis_adaxial_thickness.std(),
                        'ABEpidermisThickness': epidermis_abaxial_thickness.mean(),
                        'ABEpidermisThickness_SD': epidermis_abaxial_thickness.std(),
                        'LeafVolume': leaf_volume,
                        'MesophyllVolume': mesophyll_volume,
                        'ADEpidermisVolume': epidermis_adaxial_volume,
                        'ABEpidermisVolume': epidermis_abaxial_volume,
                        'VeinVolume': vein_volume,
                        'BSVolume': bs_volume,
                        'VeinBSVolume': vein_volume+bs_volume,
                        'CellVolume': cell_volume,
                        'IASVolume': air_volume,
                        'IASSurfaceArea': true_ias_SA,
                        'PixelSize': px_edge,
                        'Units': units,
                        '_SLICEStrimmed':trim_slices,
                        '_X_trimmed_left':trim_column_L*to_resize,
                        '_X_trimmed_right':trim_column_L*to_resize}
                        #            'IASLargestConnectedVolume':largest_connected_ias_volume,
                        #            'IASLargestConnectedSA':largest_connected_ias_SA

            results_out = DataFrame(data_out, index={sample_name})
            # Save the data to a CSV
            print('### Saving results to text file ###')
            results_out.to_csv(base_folder_name + sample_name + '/' + sample_name
                               + 'RESULTS.txt', sep='\t', encoding='utf-8')
            print('Data saved')
            for key, value in data_out.items():
                if isinstance(value, str):
                    print(str(key) + ' : ' + value)
                else:
                    print(str(key) + ' : ' + str(round(value, 3)))
            print('')
            print('Done with ' + sample_name)
            print('')
        #
            #
        else:
            print('\nNot all required arguments are defined. Check command line input and try again.\n')
            print(sample_name, str(px_edge), to_resize, units, reuse_raw_binary, str(trim_slices), color_values, binary_postfix, base_folder_name)


if __name__ == '__main__':
    main()
