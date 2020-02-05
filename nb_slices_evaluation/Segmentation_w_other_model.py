#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 16 10:07:35 2018

@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""

# NOTE FROM GTR, THE LAST CONTRIBUTOR TO THIS CODE.
# This code is intended for a non-interactive command line use.
# If you wish to use it interactively, uncomment lines 20-29 and edit
# accordingly based on your sample.

# FOR THE FIRST TIME YOU USE THIS CODE
# Edit lines 75-77 in order to reflect fyour file naming convention.

# Import libraries
from Leaf_Segmentation_Functions_py3 import *  # Custom ML microCT functions
import sys
import os
import gc
from sklearn.externals import joblib
import zipfile

# Extract data from command line input
full_script_path = sys.argv[0]
sample_name = sys.argv[1]
raw_slices = sys.argv[4]
rescale_factor = int(sys.argv[5])
threshold_rescale_factor = int(sys.argv[6])
base_folder_name = sys.argv[7]
model_path = sys.argv[8]

# Set directory of functions in order to import MLmicroCTfunctions
path_to_script = '/'.join(full_script_path.split('/')[:-1]) + '/'
# os.chdir(path_to_script)

# Get the slice numbers into a vector of integer
ImgJ_slices = [int(x) for x in raw_slices.split(',')]

# Define the values of each tissue in the labelled stacks
# Note: Not sure this is needed anymore - Need to check in the functions.
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152
bs_value = 128

postfix_phase = 'PAGANIN-8bit.tif'
postfix_grid = 'GRID-8bit.tif'

# Create folder and define file names to be used
folder_name = sample_name + '/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')
from sklearn.externals import joblib
filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = sample_name + postfix_grid #'GRID-8bit.tif'
phase_name = sample_name + postfix_phase #'PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# Below takes the slices from imageJ notation, put them in python notation
# (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
# of the same length, shuffle that sequence, and shuffle the slices in the same
# order. This creates a bit of randomness in the training-testing slices.
labelled_slices = np.array(ImgJ_slices) - 1

if os.path.isfile(filepath+grid_name+'_2x-smaller.tif') == False:
    if os.path.isfile(filepath+grid_name+'_2x-smaller.tif'+'.zip'):
        with zipfile.ZipFile(filepath+grid_name+'_2x-smaller.tif'+'.zip', 'r') as zip_ref:
            zip_ref.extractall(filepath+grid_name)

if os.path.isfile(filepath+phase_name+'_2x-smaller.tif') == False:
    if os.path.isfile(filepath+phase_name+'_2x-smaller.tif'+'.zip'):
        with zipfile.ZipFile(filepath+phase_name+'_2x-smaller.tif'+'.zip', 'r') as zip_ref:
            zip_ref.extractall(filepath+phase_name)

# Load the images
print('***LOADING IMAGES***')
gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(
    filepath, phase_name, rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+sample_name+'local_thick.tif'):
    print('***LOADING LOCAL THICKNESS***')
    localthick_stack = localthick_load_and_resize(
        folder_name, sample_name, threshold_rescale_factor)
elif os.path.isfile(folder_name+'local_thick.tif'):
    print('***RENAMING FILE FROM OLDER VERSION OF CODE***')
    os.rename(folder_name+'local_thick.tif', folder_name+sample_name+'local_thick.tif')
    print('***LOADING LOCAL THICKNESS***')
    localthick_stack = localthick_load_and_resize(
        folder_name, sample_name, threshold_rescale_factor)

print('***LOADING TRAINED MODEL***')
rf_transverse = joblib.load(model_path)
print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
    oob=rf_transverse.oob_score_ * 100)))

print('***STARTING FULL STACK PREDICTION***')
RFPredictCTStack_out = RFPredictCTStack(
    rf_transverse, gridrec_stack[labelled_slices,:,:], phaserec_stack[labelled_slices,:,:],
     localthick_stack[labelled_slices,:,:], "transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+sample_name+"prediction_w_other_model.tif", RFPredictCTStack_out)
         # img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))
print('Done!')
