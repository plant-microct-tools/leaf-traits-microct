#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-08-20

@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""

import os
import numpy as np
import pandas as pd
import skimage.io as io
from tabulate import tabulate

def performance_metrics(stack,gp_test_slices,label_stack,label_test_slices,folder_name,sample_name):
    # e.g.
    # tag = "\nPost-processed Full Stack Scores:\n"
    # performance_metrics(processed,gridphase_test_slices_subset,label_stack,label_test_slices_subset,folder_name,tag)
    # generate absolute confusion matrix
    conf_matrix = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"),rownames=['Actual'], colnames=['Predicted'])
    conf_matrix.to_csv(folder_name+'Results/'+sample_name+'_ConfusionMatrix.txt',header='Predicted', index='Actual', sep='\t', mode='w')
    # generate normalized confusion matrix
    conf_matrix_norm = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"), rownames=['Actual'], colnames=['Predicted'], normalize='index')
    conf_matrix_norm.to_csv(folder_name+'Results/'+sample_name+'_NormalizedConfusionMatrix.txt',header='Predicted', index='Actual', sep='\t', mode='w')
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
    results_out.to_csv(folder_name+'Results/'+sample_name+'_PerformanceMetrics.txt', sep='\t', encoding='utf-8')

    file = open(folder_name+'Results/'+sample_name+'_PerformanceMetrics.txt', 'a')
    file.write("\n")
    file.write("\nTotal accuracy: "+str(total_accuracy*100)+"%\n")
    file.write("Total test pixels: "+str(total_testpixels))

    return conf_matrix, precision, recall

def print_feature_layers(rf_t,folder_name):
    # Print feature layer importance
    file = open('../results/'+folder_name+'/FeatureLayer.txt','w')
    file.write('Our OOB prediction of accuracy for is: {oob}%'.format(oob=rf_t.oob_score_ * 100)+'\n')
    feature_layers = range(0,len(rf_t.feature_importances_))
    for fl, imp in zip(feature_layers, rf_t.feature_importances_):
        file.write('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp)+'\n')
    file.close()

# DEFINE SOME NAMES
folder_name = '/Users/guillaumetherouxrancourt/Dropbox/PostDoc/Projects/LeafTrait_MachineLearning/Test_Data_for_Manuscript/'
filename = 'C_I_12_Strip2_fullstack_prediction_train_2_test_8.tif'
name_split = filename[:-4].split('_') # Remove file extension and split
sample_name = '_'.join(name_split[0:4])
test_name = '_'.join(name_split[0:4] + name_split[6:10])
train_slices = list(map(int, name_split[7].split('-')))
test_slices = list(map(int, name_split[9].split('-')))

# CHECK IF RESULTS FOLDER EXISTS
if os.path.exists(folder_name + 'Results/') == False:
    os.makedirs(folder_name + 'Results/')

# LOAD HAND LABELLED STACK
labelled_stack = io.imread(folder_name + 'labelled-stack.tif_2x-smaller.tif')
print(np.unique(labelled_stack))
cell_lbl, epid_lbl, bs_lbl, vein_lbl, bg_lbl, air_lbl = np.unique(labelled_stack)

# LABELLED SLICES ON ORIGINAL STACK
labelled_slices = np.array([80,140,200,260,340,400,440,540,620,740,800,860,940,1060,1140,1240,1300,1400,1480,1540,1600,1690,1740,1840])-1
labelled_slices_seq = np.arange(len(labelled_slices))

# LOAD PREDICTED STACK
pred_stack = io.imread(folder_name + 'NB_slices_eval/C_I_12_Strip2_fullstack_prediction_train_2_test_8.tif')
print(np.unique(pred_stack))
if np.any(np.unique(pred_stack) < 0):
    pred_stack = np.where(pred_stack < 0, pred_stack + 256, pred_stack)
print(np.unique(pred_stack))

cell_pred, epid_pred, bs_pred, vein_pred, bg_pred, air_pred = np.unique(pred_stack)

# CHANGE COLOR OF PREDICTED STACK
pred_stack = np.where(pred_stack == epid_pred, epid_lbl, pred_stack)
pred_stack = np.where(pred_stack == bs_pred, bs_lbl, pred_stack)
pred_stack = np.where(pred_stack == vein_pred, vein_lbl, pred_stack)
pred_stack = np.where(pred_stack == bg_pred, bg_lbl, pred_stack)

# COMPARE LABELLED VS. PREDICTED
indices = [x for x in labelled_slices_seq if x != train_slices]
conf_matrix, precision, recall = performance_metrics(pred_stack,indices,labelled_stack,indices,folder_name,test_name)

# LOAD MODEL
if os.path.isfile(folder_name+sample_name+'RF_model.joblib'):
    print('***LOADING TRAINED MODEL***')
    rf_transverse = joblib.load(folder_name+sample_name+'RF_model.joblib')
    print(('Our Out Of Box prediction of accuracy is: {oob}%'.format(
        oob=rf_transverse.oob_score_ * 100)))


    if pred_size == 'label':
        gp_test_slices = label_test_slices
    elif pred_size == 'min-max':

# OOB predictions
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
