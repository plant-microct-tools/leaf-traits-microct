
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thurs Oct 31 17:11:35 2019
@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""

# This code is intended for a non-interactive command line use.
# New functions added to 'Leaf_Segmentation_Functions_py3.py' by MJ

# Import libraries
from Leaf_Segmentation_Functions_py3 import *  # Custom ML microCT functions
import sys
import os
import gc
from sklearn.externals import joblib

def main():
    # Extract data from command line input
    path = sys.argv[0]
    argfiles = sys.argv[1]

    print(sys.argv)
    print(len(sys.argv))

    # if len(sys.argv) > 2:
    #     for ii in range(2, len(sys.argv)):
    #         print(sys.argv[ii])
    #         exec(sys.argv[ii])

    path_to_argfile_folder = '/'.join(path.split('/')[:-1]) + '/argfile_folder/'

    # define some things
    j = 0
    permission = 0
    filenames = []
    for z in argfiles.split(','):
        z.strip()
        z = z.replace('\n','')
        filenames.append(z)

    # optional, but nice catch for incorrect filepath or filename entry
    for i in range(0,len(filenames)):
        if os.path.exists(path_to_argfile_folder+filenames[i]) == False:
            print("\nSome of the information you entered is incorrect. Try again.\n")
            permission = 1

    while j < len(filenames) and permission == 0:
        print('\nWorking on scan: '+str(j+1)+' of '+str(len(filenames))+'\n')
        #read input file and define lots of stuff
        list_of_lines = openAndReadFile(path_to_argfile_folder+filenames[j])
        print(list_of_lines)

        var_names = ['sample_name', 'postfix_phase', 'Th_phase', 'postfix_grid', 'Th_grid', 'nb_training_slices',
                     'raw_slices', 'rescale_factor', 'threshold_rescale_factor', 'nb_estimators', 'base_folder_name']
        

        sample_name=postfix_phase=Th_phase=postfix_grid=Th_grid=nb_training_slices=raw_slices=rescale_factor=threshold_rescale_factor=nb_estimators=base_folder_name = str()

        # If input have be added to the command line, there will be more than 2 arguments, so below won't affect
        # when all input have been entered to the arg file.
        sys_arg_nb = 2
        sample_name, sys_arg_nb = check_if_empty(list_of_lines[0], sys_arg_nb)
        postfix_phase, sys_arg_nb = check_if_empty(list_of_lines[1], sys_arg_nb)
        Th_phase, sys_arg_nb = check_if_empty(list_of_lines[2], sys_arg_nb)
        postfix_grid, sys_arg_nb = check_if_empty(list_of_lines[3], sys_arg_nb)
        Th_grid, sys_arg_nb = check_if_empty(list_of_lines[4], sys_arg_nb)
        nb_training_slices, sys_arg_nb = check_if_empty(list_of_lines[5], sys_arg_nb)
        raw_slices, sys_arg_nb = check_if_empty(list_of_lines[6], sys_arg_nb)
        rescale_factor, sys_arg_nb = check_if_empty(list_of_lines[7], sys_arg_nb)
        threshold_rescale_factor, sys_arg_nb = check_if_empty(list_of_lines[8], sys_arg_nb)
        nb_estimators, sys_arg_nb = check_if_empty(list_of_lines[9], sys_arg_nb)
        base_folder_name, sys_arg_nb = check_if_empty(list_of_lines[10], sys_arg_nb)

        for ii in range(len(var_names)):
            print(var_names[ii])
            print(locals()[var_names[ii]])

        Th_phase = int(Th_phase)
        Th_grid = int(Th_grid)
        nb_training_slices = int(nb_training_slices)
        rescale_factor = int(rescale_factor)
        threshold_rescale_factor = int(threshold_rescale_factor)
        nb_estimators = int(nb_estimators)

        # define parameters using list_of_lines
        # sample_name, postfix_phase, Th_phase, postfix_grid, Th_grid, nb_training_slices, raw_slices, rescale_factor, threshold_rescale_factor, nb_estimators, base_folder_name = define_params(
        #     list_of_lines)

        # Get the slice numbers into a vector of integer
        print(raw_slices)
        ImgJ_slices = [int(x) for x in raw_slices.split(',')]

        # Create folder and define file names to be used
        folder_name = sample_name + '/'
        if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
            os.makedirs(base_folder_name + folder_name + 'MLresults/')
        filepath = base_folder_name + folder_name
        folder_name = filepath + 'MLresults/'
        grid_name = sample_name + postfix_grid #'GRID-8bit.tif'
        phase_name = sample_name + postfix_phase #'PAGANIN-8bit.tif'
        label_name = 'labelled-stack.tif'
# TESTING
        # print(grid_name)
        # print(phase_name)
        # print(label_name)
# TESTING
        # Below takes the slices from imageJ notation, put them in python notation
        # (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
        # of the same length, shuffle that sequence, and shuffle the slices in the same
        # order. This creates a bit of randomness in the training-testing slices.
        labelled_slices = np.array(ImgJ_slices) - 1
        labelled_slices_seq = np.arange(labelled_slices.shape[0])
        np.random.shuffle(labelled_slices_seq)
        labelled_slices = labelled_slices[labelled_slices_seq]
        # Set the training slices to be at least one more slices than the test slices.
        train_slices = np.arange(0, stop=nb_training_slices)
        test_slices = np.arange(len(train_slices), stop=len(labelled_slices))
# TESTING
        # print(train_slices)
        # print(test_slices)
# TESTING
#
        # Load the images
        print('***LOADING IMAGES***')
        gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
        phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
        label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor, labelled_stack=True)
        if len(label_stack.shape) == 4:
            label_stack = label_stack[:,:,:,0]
#
#
        # Load the stacks and downsize a copy of the binary to speed up the thickness processing
        if os.path.isfile(folder_name+'/'+sample_name+'GridPhase_invert_ds.tif') == False:
            print('***CREATE THE THRESHOLDED IMAGE***')
            Threshold_GridPhase_invert_down(gridrec_stack, phaserec_stack, Th_grid, Th_phase, folder_name,
            sample_name, threshold_rescale_factor)

        # Generate the local thickness
        if os.path.isfile(folder_name+sample_name+'local_thick.tif'):
            print('***LOADING LOCAL THICKNESS***')
            localthick_stack = localthick_load_and_resize(folder_name, sample_name, threshold_rescale_factor)
        else:
            print('***GENERATE LOCAL THICKNESS***')
            localthick_up_save(folder_name, sample_name, keep_in_memory=False)
            localthick_stack = localthick_load_and_resize(folder_name, sample_name, threshold_rescale_factor)
#
        #define image subsets for training and testing
        gridphase_train_slices_subset = labelled_slices[train_slices]
        gridphase_test_slices_subset = labelled_slices[test_slices]
        label_train_slices_subset = labelled_slices_seq[train_slices]
        label_test_slices_subset = labelled_slices_seq[test_slices]
# TESTING
        # print(gridphase_train_slices_subset)
        # print(gridphase_test_slices_subset)
        # print(label_train_slices_subset)
        # print(label_test_slices_subset)
# TESTING
#
        print("")
        displayImages_displayDims(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                          gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset)
        print("")
#
#
        if os.path.isfile(folder_name+sample_name+'RF_model.joblib'):
            print('***LOADING TRAINED MODEL***')
            rf_transverse = joblib.load(folder_name+sample_name+'RF_model.joblib')
            print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
            oob=rf_transverse.oob_score_ * 100)))
        else:
            print('***STARTING MODEL TRAINING***')
            print(("    Training slices (" + str(len(train_slices))+ " slices):" + str(gridphase_train_slices_subset)))
            print(("    Test slices (" + str(len(test_slices))+ " slices):" + str(gridphase_test_slices_subset)))

            rf_transverse = train_model(gridrec_stack, phaserec_stack, label_stack, localthick_stack,
                                gridphase_train_slices_subset, gridphase_test_slices_subset,
                                label_train_slices_subset, label_test_slices_subset, nb_estimators)

            print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
                oob=rf_transverse.oob_score_ * 100)))
            gc.collect()
            print('***SAVING TRAINED MODEL***')
            joblib.dump(rf_transverse, folder_name+sample_name+'RF_model.joblib',compress='zlib')

        print('***STARTING FULL STACK PREDICTION***')
        RFPredictCTStack_out = RFPredictCTStack(rf_transverse, gridrec_stack, phaserec_stack, localthick_stack, "transverse")
        joblib.dump(RFPredictCTStack_out, folder_name+sample_name+'RFPredictCTStack_out.joblib',compress='zlib')
        io.imsave(folder_name+sample_name+"fullstack_prediction.tif", RFPredictCTStack_out)
        print('Done!')
#
        j = j + 1

if __name__ == '__main__':
    main()
