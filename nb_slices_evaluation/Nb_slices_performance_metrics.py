#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-08-20

@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""

import os
# import joblib
import numpy as np
import pandas as pd
import skimage.io as io
import sys
from tabulate import tabulate

def performance_metrics(stack,gp_test_slices,label_stack,label_test_slices,folder_name,sample_name):
    # e.g.
    # tag = "\nPost-processed Full Stack Scores:\n"
    # performance_metrics(processed,gridphase_test_slices_subset,label_stack,label_test_slices_subset,folder_name,tag)
    # generate absolute confusion matrix
    conf_matrix = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"),rownames=['Actual'], colnames=['Predicted'])
    conf_matrix.to_csv(folder_name+sample_name+'_ConfusionMatrix.txt',header='Predicted', index='Actual', sep='\t', mode='w')
    # generate normalized confusion matrix
    conf_matrix_norm = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"), rownames=['Actual'], colnames=['Predicted'], normalize='index')
    conf_matrix_norm.to_csv(folder_name+sample_name+'_NormalizedConfusionMatrix.txt',header='Predicted', index='Actual', sep='\t', mode='w')
    print(tabulate(conf_matrix_norm, headers='keys', tablefmt='pqsl'))
    # total acccuracy
    total_testpixels = stack.shape[1]*stack.shape[2]*len(gp_test_slices)
    total_accuracy = float(np.diag(conf_matrix).sum()) / total_testpixels
    print("\nTotal accuracy is: "+str(total_accuracy*100)+"%\n")
    precision = np.diag(conf_matrix)/np.sum(conf_matrix,1)
    recall = np.diag(conf_matrix)/np.sum(conf_matrix,0)
    f1_score = 2 * ((precision*recall)/(precision+recall))
    print(precision)
    print(recall)
    print(f1_score)
    data_out = {'precision': precision,
                'recall': recall,
                'f1_score': f1_score}
    results_out = pd.DataFrame(data_out)
    results_out.to_csv(folder_name+sample_name+'_PerformanceMetrics.txt', sep='\t', encoding='utf-8')
    file = open(folder_name+sample_name+'_PerformanceMetrics.txt', 'a')
    file.write("\n")
    file.write("\nTotal accuracy: "+str(total_accuracy*100)+"%\n")
    file.write("Total test pixels: "+str(total_testpixels))
    file.close()
    return conf_matrix, precision, recall

def print_feature_layers(rf_t,folder_name):
    # Print feature layer importance
    file = open('../results/'+folder_name+'/FeatureLayer.txt','w')
    file.write('Our OOB prediction of accuracy for is: {oob}%'.format(oob=rf_t.oob_score_ * 100)+'\n')
    feature_layers = range(0,len(rf_t.feature_importances_))
    for fl, imp in zip(feature_layers, rf_t.feature_importances_):
        file.write('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp)+'\n')
    file.close()

# GET THE DATA FROM THE COMMAND LINE INPUT
full_script_path = sys.argv[0]
filename = sys.argv[1]
labelled_name = sys.argv[2]
raw_slices = sys.argv[3]
path = os.getcwd()
file_w_path = os.path.realpath(filename)
filesize = os.path.getsize(file_w_path)

# WELCOME MESSAGE
print('\n###########################')
print('###    STARTING WITH    ###')
print(filename)
print('\n')

# LABELLED SLICES ON ORIGINAL STACK
labelled_slices = np.array([int(x) for x in raw_slices.split(',')]) - 1
labelled_slices_seq = np.arange(len(labelled_slices))

# LOAD PREDICTED STACK
pred_stack = io.imread(file_w_path)

# STOP IF THIS IS A FULL STACK PREDICTION
# WILL DEAL WITH THAT LATER
if pred_stack.shape[0] > 24:
    correction = 80  # ADD LOGICAL FOR WHEN IT'S ONLY IN THE MIN-MAX RANGE
    print('### FILE IS A FULLSTACK PREDICTION ###')
    print('### KEEPING ONLY THE LABELLED SLCS ###')
    pred_stack = pred_stack[labelled_slices-correction, :, :]
    print(pred_stack.shape)

# DEFINE SOME NAMES
folder_name = path + '/Results/'
name_split = filename[:-4].split('_') # Remove file extension and split
sample_name = '_'.join(name_split[0:4])
test_name = '_'.join(name_split[0:4] + name_split[6:10])
train_slices = list(map(int, name_split[7].split('-')))
test_slices = list(map(int, name_split[9].split('-')))
model_filename = '_'.join(name_split[0:4]) + '_RF_model_' + '_'.join(name_split[6:10]) + '.joblib'

# CHECK IF RESULTS FOLDER EXISTS
if os.path.exists(folder_name) == False:
    os.makedirs(folder_name)

# LOAD HAND LABELLED STACK
labelled_stack = io.imread(path + '/' + labelled_name)
print(np.unique(labelled_stack))
cell_lbl, epid_lbl, bs_lbl, vein_lbl, bg_lbl, air_lbl = np.unique(labelled_stack)

# CHECK FOR LABELLED VALUES
print(np.unique(pred_stack))
if np.any(np.unique(pred_stack) < 0):
    pred_stack = np.where(pred_stack < 0, pred_stack + 256, pred_stack)
print(np.unique(pred_stack))

# CHANGE COLOR OF PREDICTED STACK
cell_pred, epid_pred, bs_pred, vein_pred, bg_pred, air_pred = np.unique(pred_stack)
pred_stack = np.where(pred_stack == epid_pred, epid_lbl, pred_stack)
pred_stack = np.where(pred_stack == bs_pred, bs_lbl, pred_stack)
pred_stack = np.where(pred_stack == vein_pred, vein_lbl, pred_stack)
pred_stack = np.where(pred_stack == bg_pred, bg_lbl, pred_stack)

# COMPARE LABELLED VS. PREDICTED
indices = [x for x in labelled_slices_seq if np.all(x != train_slices)]
conf_matrix, precision, recall = performance_metrics(pred_stack, indices, labelled_stack, indices, folder_name, test_name)

# # LOAD MODEL
# rf_transverse = joblib.load(folder_name + model_filename)
#     print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
#         oob=rf_transverse.oob_score_ * 100)))
#
#
#     if pred_size == 'label':
#         gp_test_slices = label_test_slices
#     elif pred_size == 'min-max':
#
# # OOB predictions
# print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
