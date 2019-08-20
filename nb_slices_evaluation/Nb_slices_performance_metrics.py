#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019-08-20

@author: Guillaume Theroux-Rancourt, Matt Jenkins, J. Mason Earles
"""


def performance_metrics(stack,gp_test_slices,label_stack,label_test_slices,folder_name,tag):
    # generate absolute confusion matrix
    conf_matrix = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"),rownames=['Actual'], colnames=['Predicted'])
    # generate normalized confusion matrix
    conf_matrix_norm = pd.crosstab(stack[gp_test_slices,:,:].ravel(order="F"),label_stack[label_test_slices,:,].ravel(order="F"), rownames=['Actual'], colnames=['Predicted'], normalize='index')
    # total acccuracy
    total_testpixels = stack.shape[1]*stack.shape[2]*len(gp_test_slices)
    total_accuracy = float(np.diag(conf_matrix).sum()) / total_testpixels
    print("\nTotal accuracy is: "+str(total_accuracy*100)+"%\n")
    precision = np.diag(conf_matrix)/np.sum(conf_matrix,1), "Precision"
    recall = np.diag(conf_matrix)/np.sum(conf_matrix,0), "Recall"
    print(precision)
    print(recall)
    if tag == "Unprocessed Full Stack Scores:\n":
        with open('../results/'+folder_name+'/PerformanceMetrics.txt', 'w') as metrics_file:
            metrics_file.truncate(0)
            metrics_file.write(tag+'\nAbsolute precision: {x}%'.format(x=total_accuracy*100)+'\n')
            metrics_file.close()
    else:
        with open('../results/'+folder_name+'/PerformanceMetrics.txt', 'a') as metrics_file:
            metrics_file.write(tag+'\nAbsolute precision: {x}%'.format(x=total_accuracy*100)+'\n')
            metrics_file.close()

def print_feature_layers(rf_t,folder_name):
    # Print feature layer importance
    file = open('../results/'+folder_name+'/FeatureLayer.txt','w')
    file.write('Our OOB prediction of accuracy for is: {oob}%'.format(oob=rf_t.oob_score_ * 100)+'\n')
    feature_layers = range(0,len(rf_t.feature_importances_))
    for fl, imp in zip(feature_layers, rf_t.feature_importances_):
        file.write('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp)+'\n')
    file.close()

def make_conf_matrix(L_test,class_p,folder_name):
    # Generate confusion matrix for transverse section
    # FIX: better format the output of confusion matrix to .txt file
    df = pd.crosstab(L_test, class_p, rownames=['Actual'], colnames=['Predicted'])
    print(tabulate(df, headers='keys', tablefmt='pqsl'))
    df.to_csv('../results/'+folder_name+'/ConfusionMatrix.txt',header='Predicted', index='Actual', sep=' ', mode='w')

def make_normconf_matrix(L_test,class_p,folder_name):
    # Generate normalized confusion matrix for transverse section
    # FIX: better format the output of confusion matrix to .txt file
    df = pd.crosstab(L_test, class_p, rownames=['Actual'], colnames=['Predicted'], normalize='index')
    print(tabulate(df, headers='keys', tablefmt='pqsl'))
    df.to_csv('../results/'+folder_name+'/NormalizedConfusionMatrix.txt',header='Predicted', index='Actual', sep=' ', mode='w')
