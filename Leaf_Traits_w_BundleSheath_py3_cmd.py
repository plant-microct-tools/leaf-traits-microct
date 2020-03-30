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
from Leaf_Segmentation_Functions_py3 import delete_dangling_epidermis, openAndReadFile, Trim_Individual_Stack, define_params_traits, tissue_cleanup_and_analysis
#import cv2

__author__ = "Guillaume Théroux-Rancourt and Matt Jenkins"
__copyright__ = ""
__credits__ = ["Guillaume Théroux-Rancourt and Matt Jenkins"]
__license__ = "MIT"
__version__ = "0.2.1"
__maintainer__ = "Guillaume Théroux-Rancourt"
__email__ = "guillaume.theroux-rancourt@boku.ac.at"
__status__ = "beta"

# Last edited by: MRJ on 2020.03.22 16:00 Pacfic

def main():
    # Extract data from command line input
    path = sys.argv[0]

    # create python-dictionary from command line inputs (ignore first) list
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
            if key == 'path_to_image_folder':
                base_folder_name = str(value)

            # if key == 'tissue_values':  # removed 02.2020; replaced with lines 81 - 134
            #     color_values = str(value)
            # new method allows user to enter up to 10 unique tissue clases
            if key == 'tissue1':
                tissue1 = str(value)
            if key == 'tissue2':
                tissue2 = str(value)
            if key == 'tissue3':
                tissue3 = str(value)
            if key == 'tissue4':
                tissue4 = str(value)
            if key == 'tissue5':
                tissue5 = str(value)
            if key == 'tissue6':
                tissue6 = str(value)
            if key == 'tissue7':
                tissue7 = str(value)
            if key == 'tissue8':
                tissue8 = str(value)
            if key == 'tissue9':
                tissue9 = str(value)
            if key == 'tissue10':
                tissue10 = str(value)

            # set up default values for some optional parameters
            try: trim_slices
            except NameError: trim_slices = 0
            try: trim_column_L
            except NameError: trim_column_L = 0
            try: trim_column_R
            except NameError: trim_column_R = 0
            try: px_edge
            except NameError: px_edge = 1


    if len(filenames)>0: # read instructions from .txt file mode, not updated for flexible tissue classes as of 02.2020
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

            # Define the dimension of a pixel
            if to_resize > 1:
                px_dimension = (px_edge, px_edge/to_resize, px_edge/to_resize)
            else:
                px_dimension = (px_edge, px_edge, px_edge)

            # Pixel dimmension
            vx_volume = px_edge**3

        # TESTING
            # print(vx_volume)
            # print(color_values)
        # TESTING

            if 'tissue1' in locals():
                t1_name, t1_value, t1_split, t1_sa, t1_step, t1_volThresh = [x for x in tissue1.split(',')]
                t1_name = str(t1_name)
                t1_value = int(t1_value)
            else: pass
            if 'tissue2' in locals():
                t2_name, t2_value, t2_split, t2_sa, t2_step, t2_volThresh = [x for x in tissue2.split(',')]
                t2_name = str(t2_name)
                t2_value = int(t2_value)
            else: pass
            if 'tissue3' in locals():
                t3_name, t3_value, t3_split, t3_sa, t3_step, t3_volThresh = [x for x in tissue3.split(',')]
                t3_name = str(t3_name)
                t3_value = int(t3_value)
            else: pass
            if 'tissue4' in locals():
                t4_name, t4_value, t4_split, t4_sa, t4_step, t4_volThresh = [x for x in tissue4.split(',')]
                t4_name = str(t4_name)
                t4_value = int(t4_value)
            else: pass
            if 'tissue5' in locals():
                t5_name, t5_value, t5_split, t5_sa, t5_step, t5_volThresh = [x for x in tissue5.split(',')]
                t5_name = str(t5_name)
                t5_value = int(t5_value)
            else: pass
            if 'tissue6' in locals():
                t6_name, t6_value, t6_split, t6_sa, t6_step, t6_volThresh = [x for x in tissue6.split(',')]
                t6_name = str(t6_name)
                t6_value = int(t6_value)
            else: pass
            if 'tissue7' in locals():
                t7_name, t7_value, t7_split, t7_sa, t7_step, t7_volThresh = [x for x in tissue7.split(',')]
                t7_name = str(t7_name)
                t7_value = int(t7_value)
            else: pass
            if 'tissue8' in locals():
                t8_name, t8_value, t8_split, t8_sa, t8_step, t8_volThresh = [x for x in tissue8.split(',')]
                t8_name = str(t8_name)
                t8_value = int(t8_value)
            else: pass
            if 'tissue9' in locals():
                t9_name, t9_value, t9_split, t9_sa, t9_step, t9_volThresh = [x for x in tissue9.split(',')]
                t9_name = str(t9_name)
                t9_value = int(t9_value)
            else: pass
            if 'tissue10' in locals():
                t10_name, t10_value, t10_split, t10_sa, t10_step, t10_volThresh = [x for x in tissue10.split(',')]
                t10_name = str(t10_name)
                t10_value = int(t10_value)
            else: pass

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

    ##### move funciton tissue_cleanup() from scratch.py to 'Leaf_Segmentation_Functions_py3.py', so it can be called easily --> done 3.23.2020 by MRJ
    ##### then use this function based on the tissues defined by the user (up to 10)
                if 'tissue1' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t1_name, t1_value, t1_split, t1_sa, t1_step, t1_volThresh, px_edge, units)
                    if os.path.isfile(filepath + sample_name + 'LEAFtraits.txt'):
                        with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                            file.write(str(sample_name)+'\n'+str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                            file.close()
                    else:
                        with open(filepath + sample_name + 'LEAFtraits.txt', 'w', encoding='utf-8') as file:
                            file.write(str(sample_name)+'\n'+str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                            file.close()
                if 'tissue2' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t2_name, t2_value, t2_split, t2_sa, t2_step, t2_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue3' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t3_name, t3_value, t3_split, t3_sa, t3_step, t3_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue4' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t4_name, t4_value, t4_split, t4_sa, t4_step, t4_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue5' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t5_name, t5_value, t5_split, t5_sa, t5_step, t5_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue6' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t6_name, t6_value, t6_split, t6_sa, t6_step, t6_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue7' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t7_name, t7_value, t7_split, t7_sa, t7_step, t7_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue8' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t8_name, t8_value, t8_split, t8_sa, t8_step, t8_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue9' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t9_name, t9_value, t9_split, t9_sa, t9_step, t9_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue10' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t10_name, t10_value, t10_split, t10_sa, t10_step, t10_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
# test
            j += 1
    else:  # single scan mode, updated for flexible tissue classes as of 02.2020
        # check for definitions of all required parameters before proceeding
        try: sample_name, px_edge, to_resize, units, reuse_raw_binary, trim_slices, base_folder_name # removed color_values 02.2020
        except NameError: req_not_def = 1
        if reuse_raw_binary=='True':
            try: binary_postfix
            except NameError: req_not_def = 1

        # Define the dimension of a pixel
        if to_resize > 1:
            px_dimension = (px_edge, px_edge/to_resize, px_edge/to_resize)
        else:
            px_dimension = (px_edge, px_edge, px_edge)

        if req_not_def==0:
            print('\nSingle scan mode...\n')

            # Pixel dimmension
            vx_volume = px_edge**3

        # TESTING
                # print(vx_volume)
                # print(color_values)
        # TESTING

            # Define the different tissue values
            if 'tissue1' in locals():
                t1_name, t1_value, t1_split, t1_sa = [x for x in tissue1.split(',')]
                t1_name = str(t1_name)
                t1_value = int(t1_value)
            else: pass
            if 'tissue2' in locals():
                t2_name, t2_value, t2_split, t2_sa = [x for x in tissue2.split(',')]
                t2_name = str(t2_name)
                t2_value = int(t2_value)
            else: pass
            if 'tissue3' in locals():
                t3_name, t3_value, t3_split, t3_sa = [x for x in tissue3.split(',')]
                t3_name = str(t3_name)
                t3_value = int(t3_value)
            else: pass
            if 'tissue4' in locals():
                t4_name, t4_value, t4_split, t4_sa = [x for x in tissue4.split(',')]
                t4_name = str(t4_name)
                t4_value = int(t4_value)
            else: pass
            if 'tissue5' in locals():
                t5_name, t5_value, t5_split, t5_sa = [x for x in tissue5.split(',')]
                t5_name = str(t5_name)
                t5_value = int(t5_value)
            else: pass
            if 'tissue6' in locals():
                t6_name, t6_value, t6_split, t6_sa = [x for x in tissue6.split(',')]
                t6_name = str(t6_name)
                t6_value = int(t6_value)
            else: pass
            if 'tissue7' in locals():
                t7_name, t7_value, t7_split, t7_sa = [x for x in tissue7.split(',')]
                t7_name = str(t7_name)
                t7_value = int(t7_value)
            else: pass
            if 'tissue8' in locals():
                t8_name, t8_value, t8_split, t8_sa = [x for x in tissue8.split(',')]
                t8_name = str(t8_name)
                t8_value = int(t8_value)
            else: pass
            if 'tissue9' in locals():
                t9_name, t9_value, t9_split, t9_sa = [x for x in tissue9.split(',')]
                t9_name = str(t9_name)
                t9_value = int(t9_value)
            else: pass
            if 'tissue10' in locals():
                t10_name, t10_value, t10_split, t10_sa = [x for x in tissue10.split(',')]
                t10_name = str(t10_name)
                t10_value = int(t10_value)
            else: pass

        # TESTING
                # print(epid_value, bg_value, mesophyll_value)
        # TESTING

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
                if 'tissue1' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t1_name, t1_value, t1_split, t1_sa, t1_step, t1_volThresh, px_edge, units)
                    if os.path.isfile(filepath + sample_name + 'LEAFtraits.txt'):
                        with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                            file.write(str(sample_name)+'\n'+str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                            file.close()
                    else:
                        with open(filepath + sample_name + 'LEAFtraits.txt', 'w', encoding='utf-8') as file:
                            file.write(str(sample_name)+'\n'+str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                            file.close()
                if 'tissue2' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t2_name, t2_value, t2_split, t2_sa, t2_step, t2_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue3' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t3_name, t3_value, t3_split, t3_sa, t3_step, t3_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue4' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t4_name, t4_value, t4_split, t4_sa, t4_step, t4_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue5' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t5_name, t5_value, t5_split, t5_sa, t5_step, t5_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue6' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t6_name, t6_value, t6_split, t6_sa, t6_step, t6_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue7' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t7_name, t7_value, t7_split, t7_sa, t7_step, t7_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue8' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t8_name, t8_value, t8_split, t8_sa, t8_step, t8_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue9' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t9_name, t9_value, t9_split, t9_sa, t9_step, t9_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
                if 'tissue10' in locals():
                    tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA = tissue_cleanup_and_analysis(raw_pred_stack, t10_name, t10_value, t10_split, t10_sa, t10_step, t10_volThresh, px_edge, units)
                    with open(filepath + sample_name + 'LEAFtraits.txt', 'a', encoding='utf-8') as file:
                        file.write(str(tissue_name)+'\n'+'Computed volume = '+str(computed_volume)+'\n'+'Computed thickness = '+str(computed_thickness)+'\n'+'Computed Surface Area = '+str(computed_SA)+'\n')
                        file.close()
    #
            print('')
            print('Done with ' + sample_name)
            print('')

        else:
            print('\nNot all required arguments are defined. Check command line input and try again.\n')
            # print(sample_name, str(px_edge), to_resize, units, reuse_raw_binary, str(trim_slices), color_values, binary_postfix, base_folder_name)

if __name__ == '__main__':
    main()
