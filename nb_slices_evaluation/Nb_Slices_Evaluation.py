#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-08-12

@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""

# CODE ADAPTED FROM Leaf_Segmentation_py3
# AS OF 2019-08-12

# Import libraries
from Leaf_Segmentation_Functions_py3 import *  # Custom ML microCT functions
import sys
import os
import gc
from sklearn.externals import joblib
from pandas import DataFrame
import re

# Extract data from command line input
full_script_path = sys.argv[0]
sample_name = sys.argv[1]
postfix_phase = sys.argv[2]
Th_phase = int(sys.argv[3])
postfix_grid = sys.argv[4]
Th_grid = int(sys.argv[5])
nb_slices = sys.argv[6]
slice_sel = sys.argv[7]
raw_slices = sys.argv[8]
rescale_factor = int(sys.argv[9])
threshold_rescale_factor = rescale_factor
base_folder_name = sys.argv[10]
pred_type = sys.argv[11]
nb_estimators = 50 if len(sys.argv) == 12 else int(sys.argv[12])

# Set directory of functions in order to import MLmicroCTfunctions
path_to_script = '/'.join(full_script_path.split('/')[:-1]) + '/'
# os.chdir(path_to_script)

# Define the number of slices to test the segmentation on
if len(nb_slices) <= 2:
    nb_training = nb_testing = int(nb_slices)
else:
    nb_slices = [int(x) for x in nb_slices.split(',')]
    nb_training = int(nb_slices[0])
    nb_testing = int(nb_slices[1])


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
if os.path.exists(base_folder_name + folder_name + 'MLresults/'+ 'Nb_slices_eval/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/'+ 'Nb_slices_eval/')
filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = sample_name + postfix_grid #'GRID-8bit.tif'
phase_name = sample_name + postfix_phase #'PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# Below takes the slices from imageJ notation, put them in python notation
# (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
# of the same length, shuffle that sequence, and shuffle the slices in the same
# order. This creates a bit of randomness in the training-testing slices.
labelled_slices_ordered = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices_ordered.shape[0])
if slice_sel == 'random':
    np.random.shuffle(labelled_slices_seq)
    labelled_slices = labelled_slices_ordered[labelled_slices_seq]
    # Set the training and testing slices
    train_slices = np.arange(0, stop=nb_training)
    test_slices = np.arange(nb_training, stop=(nb_training + nb_testing))
    #define image subsets for training and testing
    gridphase_train_slices_subset = labelled_slices[train_slices]
    gridphase_test_slices_subset = labelled_slices[test_slices]
    label_train_slices_subset = labelled_slices_seq[train_slices]
    label_test_slices_subset = labelled_slices_seq[test_slices]
elif len(slice_sel) <= 2:
    train_slices = [int(slice_sel)]
    gridphase_train_slices_subset = labelled_slices_ordered[train_slices]
    label_train_slices_subset = labelled_slices_seq[train_slices]
    np.random.shuffle(labelled_slices_seq)
    test_slices = [labelled_slices_seq[0]]
    gridphase_test_slices_subset = labelled_slices_ordered[test_slices]
    label_test_slices_subset = test_slices
elif len(slice_sel > 2):
    slice_choice = [int(x) for x in slice_sel.split(',')]
    train_slices = slice_choice[0:(nb_training+1)]
    test_slices = slice_choice[(nb_training+1):]
    gridphase_train_slices_subset = labelled_slices_ordered[train_slices]
    gridphase_test_slices_subset = labelled_slices_ordered[test_slices]
    label_train_slices_subset = labelled_slices_seq[train_slices]
    label_test_slices_subset = labelled_slices_seq[test_slices]

# #define image subsets for training and testing
# gridphase_train_slices_subset = labelled_slices[train_slices]
# gridphase_test_slices_subset = labelled_slices[test_slices]
# label_train_slices_subset = labelled_slices_seq[train_slices]
# label_test_slices_subset = labelled_slices_seq[test_slices]

# Debugging code to check how many slices in each set
# print(train_slices)
# print(gridphase_train_slices_subset)
# print(label_train_slices_subset)
# print(test_slices)
# print(gridphase_test_slices_subset)
# print(label_test_slices_subset)
# print(len(labelled_slices))

# Load the images
print('***LOADING IMAGES***')
gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(
    filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(
    filepath, label_name, rescale_factor, labelled_stack=True)
if len(label_stack.shape) == 4:
    label_stack = label_stack[:,:,:,0]

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

# Creating training and testing string for file naming
train_test_string = '_train_'+'-'.join(map(str, label_train_slices_subset))+'_test_'+'-'.join(map(str, label_test_slices_subset))


print("")
displayImages_displayDims(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                          gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset)
print("")

# if os.path.isfile(folder_name+sample_name+'RF_model.joblib'):
#     print('***LOADING TRAINED MODEL***')
#     rf_transverse = joblib.load(folder_name+sample_name+'RF_model.joblib')
#     print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
#         oob=rf_transverse.oob_score_ * 100)))
# else:
    #rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('***STARTING MODEL TRAINING***')
print(("    Training slices (" + str(len(train_slices))
      + " slices):" + str(gridphase_train_slices_subset)))
print(("    Test slices (" + str(len(test_slices))
      + " slices):" + str(gridphase_test_slices_subset)))

rf_transverse = train_model(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                            gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset, nb_estimators)

print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
    oob=rf_transverse.oob_score_ * 100)))
gc.collect()
print('***SAVING TRAINED MODEL***')
joblib.dump(rf_transverse, folder_name+'Nb_slices_eval/'+sample_name+'RF_model'+ train_test_string +'.joblib',
            compress='zlib')
    # print_feature_layers(rf_transverse,folder_name)
    #%reset_selective -f FL_[a-z]
    #%reset_selective -f Label_[a-z]

# Make predictions on slices within the labelled slices, not at the edges.
# This would save time.

if pred_type == "full":
    pred_slices = np.arange(0, gridrec_stack.shape[0])
elif pred_type == "min-max":
    pred_slices = np.arange(labelled_slices_ordered[0],labelled_slices_ordered[-1]+1)
elif bool(re.search('label', pred_type)):
    pred_slices = labelled_slices_ordered
else:
    print('************************************************')
    raise ValueError('not a valid choice for stack prediction')

print('***STARTING STACK PREDICTION***')
RFPredictCTStack_out = RFPredictCTStack(
    rf_transverse, gridrec_stack[pred_slices,:,:], phaserec_stack[pred_slices,:,:], localthick_stack[pred_slices,:,:], "transverse")
# joblib.dump(RFPredictCTStack_out, folder_name+sample_name+'RFPredictCTStack_out.joblib',
#                     compress='zlib')
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+'Nb_slices_eval/'+sample_name+"fullstack_prediction"+train_test_string+".tif", RFPredictCTStack_out)
         # img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))

# Write to a file what has been Done
if os.path.isfile(folder_name+'Nb_slices_eval/'+'Nb_slices_evaluation_sets.txt'):
    sets = open(folder_name+'Nb_slices_eval/'+'Nb_slices_evaluation_sets.txt', 'a')
else:
    sets = open(folder_name+'Nb_slices_eval/'+'Nb_slices_evaluation_sets.txt', 'w')

if isinstance(train_slices, int):
    str_train_slices = str(label_train_slices_subset)
else:
    str_train_slices = '-'.join(map(str,label_train_slices_subset))

if isinstance(test_slices, int):
    str_test_slices = str(label_test_slices_subset)
else:
    str_test_slices = '-'.join(map(str,label_test_slices_subset))

sets.write('Nb_training: '+str(nb_training)+';; Training '+str_train_slices+';; Nb_testing: '+str(nb_testing)+';; Testing '+str_test_slices+"\n")
sets.close()

print('Done for ' + train_test_string)
