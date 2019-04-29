#!/usr/bin/env python2
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
from Leaf_Segmentation_Functions import *  # Custom ML microCT functions
import sys
import os
import gc
from sklearn.externals import joblib

## Extract data from command line input
#sample_name = 'name_'
#Th_phase = 111
#Th_grid = 122
#raw_slices = '1,2,3,4'
#rescale_factor = 2
#threshold_rescale_factor = 2
#base_folder_name = '/path/to/your/samples/directory/'
#
## Set directory of functions in order to import MLmicroCTfunctions
#os.chdir('path/to/script/and/functions/')


# Extract data from command line input
full_script_path = sys.argv[0]
sample_name = sys.argv[1]
Th_phase = int(sys.argv[2])
Th_grid = int(sys.argv[3])
raw_slices = sys.argv[4]
rescale_factor = int(sys.argv[5])
threshold_rescale_factor = int(sys.argv[6])
base_folder_name = sys.argv[7]
nb_estimators = 50 if len(sys.argv) == 8 else int(sys.argv[8])

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

# Create folder and define file names to be used
folder_name = sample_name + '/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = sample_name + 'GRID-8bit.tif'
phase_name = sample_name + 'PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# Below takes the slices from imageJ notation, put them in python notation
# (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
# of the same length, shuffle that sequence, and shuffle the slices in the same
# order. This creates a bit of randomness in the training-testing slices.
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
# Set the training slices to be at least one more slices than the test slices.
# The last part will add 1 to even length labelled slices number, and 0 to even.
# This meanes than odd length will have one training slice more, and even will have two more.
# int(len(labelled_slices) % 2 == 0))
train_slices = np.arange(0, stop=int(np.ceil(len(labelled_slices)/2)) + 1)
test_slices = np.arange(len(train_slices), stop=len(labelled_slices))

# Debugging code to check how many slices in each set
# print(train_slices)
# print(test_slices)
# print(len(labelled_slices))

# Load the images
print('***LOADING IMAGES***')
gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(
    filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(
    filepath, label_name, rescale_factor, labelled_stack=True)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/'+sample_name+'GridPhase_invert_ds.tif') == False:
    print('***CREATE THE THRESHOLDED IMAGE***')
    Threshold_GridPhase_invert_down(gridrec_stack, phaserec_stack, Th_grid,
                                    Th_phase, folder_name, sample_name, threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+sample_name+'local_thick.tif'):
    print('***LOADING LOCAL THICKNESS***')
    localthick_stack = localthick_load_and_resize(
        folder_name, sample_name, threshold_rescale_factor)
else:
    print('***GENERATE LOCAL THICKNESS***')
    localthick_up_save(folder_name, sample_name, keep_in_memory=False)
    localthick_stack = localthick_load_and_resize(
        folder_name, sample_name, threshold_rescale_factor)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

print("")
displayImages_displayDims(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                          gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset)
print("")

if os.path.isfile(folder_name+sample_name+'RF_model.joblib'):
    print('***LOADING TRAINED MODEL***')
    rf_transverse = joblib.load(folder_name+sample_name+'RF_model.joblib')
    print('Our Out Of Box prediction of accuracy is: {oob}%'.format(
        oob=rf_transverse.oob_score_ * 100))
else:
    #rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
    print('***STARTING MODEL TRAINING***')
    print("    Training slices (" + str(len(train_slices))
          + " slices):" + str(gridphase_train_slices_subset))
    print("    Test slices (" + str(len(test_slices))
          + " slices):" + str(gridphase_test_slices_subset))

    rf_transverse = train_model(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                                gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset, nb_estimators)

    print('Our Out Of Box prediction of accuracy is: {oob}%'.format(
        oob=rf_transverse.oob_score_ * 100))
    gc.collect()
    print('***SAVING TRAINED MODEL***')
    joblib.dump(rf_transverse, folder_name+sample_name+'RF_model.joblib',
                compress='zlib')
    # print_feature_layers(rf_transverse,folder_name)
    #%reset_selective -f FL_[a-z]
    #%reset_selective -f Label_[a-z]

print('***STARTING FULL STACK PREDICTION***')
RFPredictCTStack_out = RFPredictCTStack(
    rf_transverse, gridrec_stack, phaserec_stack, localthick_stack, "transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+sample_name+"fullstack_prediction.tif",
          img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))
print('Done!')
