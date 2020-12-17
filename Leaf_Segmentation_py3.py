
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 06 09:03:25 2019
@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""
# Last edited by MJ

# Import libraries
from Leaf_Segmentation_Functions_py3 import *  # Custom ML microCT functions
import sys
import os
import gc
import joblib

def main():
    # Extract data from command line input
    path = sys.argv[0]

    # create python-dictionary from command line inputs (ignore first)
    sys.argv = sys.argv[1:]
    arg_dict = dict(j.split('=') for j in sys.argv)

    # define variables for later on
    filenames = [] # list of arg files
    req_not_def = 0 # permission variable for required definitions when using command line option

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
            if key == 'phase_filename':
                postfix_phase = str(value)
            if key == 'threshold_phase':
                Th_phase = int(value)
            if key == 'grid_filename':
                postfix_grid = str(value)
            if key == 'threshold_grid':
                Th_grid = int(value)
            if key == 'nb_training_slices':
                nb_training_slices = int(value)
            if key == 'slice_numbers_training_slices':
                raw_slices = str(value)
            if key == 'rescale_factor':
                rescale_factor = int(value)
            if key == 'threshold_rescale_factor':
                threshold_rescale_factor = int(value)
            if key == 'nb_estimators':
                nb_estimators = int(value)
            if key == 'path_to_image_folder':
                base_folder_name = str(value)
            # set up default values for some optional parameters
            try: rescale_factor
            except NameError: rescale_factor = 1
            try: threshold_rescale_factor
            except NameError: threshold_rescale_factor = 1
            try: nb_estimators
            except NameError: nb_estimators = 50

    if len(filenames)>0:
        j = 0
        permission = 0

        # # check if user defined a specific argfile_folder location, otherwise default to searching in base folder
        # try: path_to_argfile_folder
        # except NameError: path_to_argfile_folder = '/'.join(path.split('/')[:-1]) + '/image_folder/'

        for i in range(0,len(filenames)): # optional but nice catch for incorrect filepath or filename entry
            if os.path.exists(path_to_argfile_folder+filenames[i]) == False:
                print('\nThe argument file is not present in the arg_file folder or path_to_argfile_folder is incorrect. Try again.\n')
                permission = 1
        while j < len(filenames) and permission == 0:
            print('\nWorking on scan: '+str(j+1)+' of '+str(len(filenames))+'\n')

            #read input file and define lots of stuff
            list_of_lines = openAndReadFile(path_to_argfile_folder+filenames[j])
            # print(list_of_lines) # comment out once built

            # define parameters using list_of_lines
            sample_name, postfix_phase, Th_phase, postfix_grid, Th_grid, nb_training_slices, raw_slices, rescale_factor, threshold_rescale_factor, nb_estimators, base_folder_name = define_params(list_of_lines)

            # Get the slice numbers into a vector of integer
            imgj_slices = [int(x) for x in raw_slices.split(',')]

            # Create folder and define file names to be used
            folder_name = sample_name + '/'
            if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
                os.makedirs(base_folder_name + folder_name + 'MLresults/')
            filepath = base_folder_name + folder_name
            folder_name = filepath + 'MLresults/'
            grid_name = sample_name + postfix_grid #'GRID-8bit.tif'
            phase_name = sample_name + postfix_phase #'PAGANIN-8bit.tif'
            label_name = 'labelled-stack.tif'

            # Below takes the slices from imageJ notation, put them in python notation
            # (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
            # of the same length, shuffle that sequence, and shuffle the slices in the same
            # order. This creates a bit of randomness in the training-testing slices.
            labelled_slices = np.array(imgj_slices) - 1
            labelled_slices_seq = np.arange(labelled_slices.shape[0])
            np.random.shuffle(labelled_slices_seq)
            labelled_slices = labelled_slices[labelled_slices_seq]
            # Set the training slices to be at least one more slices than the test slices.
            # The last part will add 1 to even length labelled slices number, and 0 to even.
            # This meanes than odd length will have one training slice more, and even will have two more.
            # int(len(labelled_slices) % 2 == 0))
            train_slices = np.arange(0, stop=nb_training_slices)
            test_slices = np.arange(len(train_slices), stop=len(labelled_slices))
    #
            # Load the images
            print('***LOADING IMAGES***')
            gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
            phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
            label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor, labelled_stack=True)
            if len(label_stack.shape) == 4:
                label_stack = label_stack[:,:,:,0]
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
    #
            print("")
            displayImages_displayDims(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                              gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset)
            print("")
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
            j += 1
    else:
        # check for definitions of all required parameters before proceeding
        try: sample_name, postfix_phase, Th_phase, postfix_grid, Th_grid, raw_slices
        except NameError: req_not_def = 1
        if req_not_def==0:
            print('\nSingle scan mode...\n')

            # Get the slice numbers into a vector of integer
            imgj_slices = [int(x) for x in raw_slices.split(',')]

            # check for definition of nb_training_slices and if not present, then define by default
            try: nb_training_slices
            except NameError: nb_training_slices = len(imgj_slices) - 1
            # check for definition of path_to_image_folder and if not present, then define by default
            try: base_folder_name
            except NameError: base_folder_name = '/'.join(path.split('/')[:-1]) + '/image_folder/'
            # print(base_folder_name)

            # Create folder and define file names to be used
            folder_name = sample_name + '/'
            if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
                os.makedirs(base_folder_name + folder_name + 'MLresults/')
            filepath = base_folder_name + folder_name
            folder_name = filepath + 'MLresults/'
            grid_name = sample_name + postfix_grid #'GRID-8bit.tif'
            phase_name = sample_name + postfix_phase #'PAGANIN-8bit.tif'
            label_name = 'labelled-stack.tif'

            # Below takes the slices from imageJ notation, put them in python notation
            # (i.e. with 0 as the first element instead of 1 in ImageJ), create a sequence
            # of the same length, shuffle that sequence, and shuffle the slices in the same
            # order. This creates a bit of randomness in the training-testing slices.
            labelled_slices = np.array(imgj_slices) - 1
            labelled_slices_seq = np.arange(labelled_slices.shape[0])
            np.random.shuffle(labelled_slices_seq)
            labelled_slices = labelled_slices[labelled_slices_seq]
            # Set the training slices to be at least one more slices than the test slices.
            # The last part will add 1 to even length labelled slices number, and 0 to even.
            # This meanes than odd length will have one training slice more, and even will have two more.
            # int(len(labelled_slices) % 2 == 0))
            train_slices = np.arange(0, stop=nb_training_slices)
            test_slices = np.arange(len(train_slices), stop=len(labelled_slices))
    #
            # Load the images
            print('***LOADING IMAGES***')
            gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
            phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
            label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor, labelled_stack=True)
            if len(label_stack.shape) == 4:
                label_stack = label_stack[:,:,:,0]
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
    #
            print("")
            displayImages_displayDims(gridrec_stack, phaserec_stack, label_stack, localthick_stack, gridphase_train_slices_subset,
                              gridphase_test_slices_subset, label_train_slices_subset, label_test_slices_subset)
            print("")
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
        else:
            print('\nNot all required arguments are defined. Check command line input and try again.\n')

if __name__ == '__main__':
    main()
