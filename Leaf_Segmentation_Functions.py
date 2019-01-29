#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:07:04 2018

@author: Matt Jenkins
"""
# All functions written by Matt Jenkins and Mason Earles unless otherwise specified.
# Functions written by Guillaume Théroux-Rancourt (GTR) will be noted so in the comments

# Import libraries
import os
import cv2
import numpy as np
import skimage.io as io
from skimage import transform, img_as_ubyte
#from skimage.external import tifffile
from skimage.filters import sobel, gaussian
#from skimage.segmentation import clear_border
from skimage.morphology import ball, remove_small_objects #, disk, cube, 
from skimage.util import invert
import scipy as sp
import scipy.ndimage as spim
#import scipy.spatial as sptl
from tabulate import tabulate
import pickle
#from PIL import Image
from tqdm import tqdm
#from numba import jit
#import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, confusion_matrix
#import matplotlib.pyplot as plt
import pandas as pd
#from scipy import misc
from scipy.ndimage.filters import maximum_filter, minimum_filter, percentile_filter
#from scipy.ndimage.morphology import distance_transform_edt
#import vtk
#import gc
# Suppress all warnings (not errors) by uncommenting next two lines of code
import warnings
warnings.filterwarnings("ignore")

# Filter parameters; Label encoder setup
disk_size=5
gauss_sd_list = [2,4,8,16,32,64] #six different filters with different sd for each, big sd = more blurred
gauss_length = 2*len(gauss_sd_list)
hess_range = [4,64]
hess_step = 4
num_feature_layers = 37 # grid and phase recon; plus gaussian blurs; plus hessian filters

# Import label encoder
labenc = LabelEncoder()

def smooth_epidermis(img,epidermis,background,spongy,palisade,ias,vein):
    # FIX: clean this up, perhaps break into multiple functions
    # Define 3D array of distances from lower and upper epidermises
    a = range(0,img.shape[1])
    b = np.tile(a,(img.shape[2],img.shape[0],1))
    b = np.moveaxis(b,[0,1,2],[2,0,1])
    # Determine the lower edge of the spongy mesophyll
    c = (img==spongy)
    d = (b*c)
    s_low = np.argmax(d, axis=1)
    s_low_adjust = np.array(s_low, copy=True)
    s_low_adjust[(s_low==img.shape[1])] = 0
    # Determine the lower edge of the palisade mesophyll
    c = (img==palisade)
    d = (b*c)
    p_low = np.argmax(d, axis=1)
    p_low_adjust = np.array(p_low, copy=True)
    p_low_adjust[(p_low==img.shape[1])] = 0
    # Determine the lower edge of the vascular bundle
    c = (img==vein)
    d = (b*c)
    v_low = np.argmax(d, axis=1)
    v_low_adjust = np.array(v_low, copy=True)
    v_low_adjust[(v_low==img.shape[1])] = 0
    # Determine the lower edge of the IAS
    c = (img==ias)
    d = (b*c)
    ias_low = np.argmax(d, axis=1)
    ias_low_adjust = np.array(ias_low, copy=True)
    ias_low_adjust[(ias_low==img.shape[1])] = 0
    # Determine the lower edge of the epidermis
    c = (img==epidermis)
    d = (b*c)
    e_low = np.argmax(d, axis=1)

    e_low = np.maximum(e_low, s_low_adjust) # Changes lowest mesophyll pixel to epidermal class
    e_low = np.maximum(e_low, p_low_adjust) # Changes lowest mesophyll pixel to epidermal class
    e_low = np.maximum(e_low, ias_low_adjust) # Changes lowest IAS pixel to epidermal class
    e_low = np.maximum(e_low, v_low_adjust) # Changes lowest vein pixel to epidermal class

    epi_low = np.zeros(img.shape)
    for z in tqdm(range(0,epi_low.shape[0])):
        for x in range(0,epi_low.shape[2]):
            epi_low[z,e_low[z,x],x] = 1

    b2 = np.flip(b,1)
    # Determine the upper edge of spongy
    c = (img==spongy)
    d = ((b2)*c)
    s_up = np.argmax(d, axis=1)
    s_up_adjust = np.array(s_up, copy=True)
    s_up_adjust[(s_up==0)] = img.shape[1]-1
    # Determine the upper edge of palisade
    c = (img==palisade)
    d = ((b2)*c)
    p_up = np.argmax(d, axis=1)
    p_up_adjust = np.array(p_up, copy=True)
    p_up_adjust[(p_up==0)] = img.shape[1]-1
    # Determine the upper edge of ias
    c = (img==ias)
    d = ((b2)*c)
    ias_up = np.argmax(d, axis=1)
    ias_up_adjust = np.array(ias_up, copy=True)
    ias_up_adjust[(ias_up==0)] = img.shape[1]-1
    # Determine the upper edge of vein
    c = (img==vein)
    d = ((b2)*c)
    v_up = np.argmax(d, axis=1)
    v_up_adjust = np.array(v_up, copy=True)
    v_up_adjust[(v_up==0)] = img.shape[1]-1
    # Determine the upper edge of epidermis
    c = (img==epidermis)
    d = ((b2)*c)
    e_up = np.argmax(d, axis=1)

    e_up = np.minimum(e_up, s_up_adjust) # Changes highest spongy pixel to epidermal class
    e_up = np.minimum(e_up, p_up_adjust) # Changes highest palisade pixel to epidermal class
    e_up = np.minimum(e_up, ias_up_adjust) # Changes highest ias pixel to epidermal class
    e_up = np.minimum(e_up, v_up_adjust) # Changes highest vein pixel to epidermal class

    epi_up = np.zeros(img.shape)
    for z in tqdm(range(0,epi_up.shape[0])):
        for x in range(0,epi_up.shape[2]):
            epi_up[z,e_up[z,x],x] = 1
    # Generate a binary stack with the pixels inside the epidermis set equal to 1
    epi_in = np.zeros(img.shape, dtype=np.uint16)
    for y in tqdm(range(0,epi_in.shape[2])):
        for z in range(0,epi_in.shape[0]):
            epi_in[z,e_up[z,y]:e_low[z,y],y] = 1
    # Generate a binary stack with the pixels outside the epidermis set equal to 1
    epi_out = (epi_in==0)*1
    # Set all background identified as IAS that lies outside epidermal boundaries as BG
    # Set all IAS identified as BG that lies within epidermal boundaries as IAS
    img2 = np.array(img, copy=True)
    img2[(img2==ias)*(epi_out==1)] = background
    img2[(img2==palisade)*(epi_out==1)] = background
    img2[(img2==spongy)*(epi_out==1)] = background
    img2[(img2==vein)*(epi_out==1)] = background
    img2[(img2==background)*(epi_in==1)] = ias

    return img2

def final_smooth(img,vein,spongy,palisade,epidermis,ias,bg):
    vein_trace = (img==vein)
    # Remove 'dangling' vein pixels
    vein_rmv_parts = np.array(vein_trace, copy=True)
    for i in tqdm(range(0,vein_rmv_parts.shape[0])):
        vein_rmv_parts[i,:,:] = remove_small_objects(vein_trace[i,:,:], min_size=600)
    # Write an array of just the removed particles
    vein_parts = vein_trace ^ vein_rmv_parts
    # Replace small vein parts with spongy mesophyll
    img[vein_parts==1] = spongy
    # Smooth veins with a double percent filter
    vein_trace_pct = np.apply_along_axis(dbl_pct_filt, 0, arr = vein_rmv_parts)
    invert_vt_pct = np.invert(vein_trace_pct)
    #Set all mesophyll identified as vein that lies oustide vein boundary as spongy mesophyll
    img4 = np.array(img, copy=True)
    img4[(img4==vein)*(invert_vt_pct==1)] = spongy
    #Set all vein identified as palisade or spongy that lies inside vein boundary as vein
    img4[(img4==palisade)*(vein_trace_pct==1)] = vein
    img4[(img4==spongy)*(vein_trace_pct==1)] = vein
    # Define 3D array of distances from lower value of img4.shape[1] to median value
    rangeA = range(0,img4.shape[1]/2)
    tileA = np.tile(rangeA,(img4.shape[2],img4.shape[0],1))
    tileA = np.moveaxis(tileA,[0,1,2],[2,0,1])
    rangeB = range(img4.shape[1]/2,img4.shape[1])
    tileB = np.tile(rangeB,(img4.shape[2],img4.shape[0],1))
    tileB = np.moveaxis(tileB,[0,1,2],[2,0,1])
    tileB = np.flip(tileB,1)
    # Define 3D array of distances from median value of img4.shape[1] to upper value
    # rangeB = range(img4.shape[1]/2,img4.shape[1])
    # tileB = np.tile(rangeB,(img4.shape[2],img4.shape[0],1))
    # tileB = np.moveaxis(tileB,[0,1,2],[2,0,1])
    # tileB = np.flip(tileB,1)
    #Make new 3d arrays of top half and lower half of image
    hold = img4.shape[1]/2
    img4conc1 = np.array(img4[:,0:hold,:], copy = True)
    img4conc2 = np.array(img4[:,hold:img4.shape[1],:], copy = True)

    # Determine the inner edge of the upper spongy
    c = (img4conc1==spongy)
    d = (tileA*c)
    s_up_in = np.argmin(d, axis=1)
    s_up_in_adjust = np.array(s_up_in, copy=True)
    s_up_in_adjust[(s_up_in==0)] = hold
    # Determine the inner edge of the upper palisade
    c = (img4conc1==palisade)
    d = (tileA*c)
    p_up_in = np.argmin(d, axis=1)
    p_up_in_adjust = np.array(p_up_in, copy=True)
    p_up_in_adjust[(p_up_in==0)] = hold
    # Determine the inner edge of the upper ias
    c = (img4conc1==ias)
    d = (tileA*c)
    ias_up_in = np.argmin(d, axis=1)
    ias_up_in_adjust = np.array(ias_up_in, copy=True)
    ias_up_in_adjust[(ias_up_in==0)] = hold
    # Determine the inner edge of the upper vein
    c = (img4conc1==vein)
    d = (tileA*c)
    v_up_in = np.argmin(d, axis=1)
    v_up_in_adjust = np.array(v_up_in, copy=True)
    v_up_in_adjust[(v_up_in==0)] = hold
    # Determine the inner edge of the upper epidermis
    c = (img4conc1==epidermis)
    d = (tileA*c)
    e_up_in = np.argmax(d, axis=1)

    e_up_in = np.minimum(e_up_in, s_up_in_adjust)
    e_up_in = np.minimum(e_up_in, p_up_in_adjust)
    e_up_in = np.minimum(e_up_in, ias_up_in_adjust)
    e_up_in = np.minimum(e_up_in, v_up_in_adjust)

    epi_up_in = np.zeros(img.shape)
    for z in range(0,epi_up_in.shape[0]):
        for x in range(0,epi_up_in.shape[2]):
            if x > 1:
                if e_up_in[z,x]==0 or e_up_in[z,x]==hold:
                    e_up_in[z,x] = e_up_in[z,x-1]
                    epi_up_in[z,e_up_in[z,x],x] = 1
                else:
                    epi_up_in[z,e_up_in[z,x],x] = 1
            else:
                epi_up_in[z,e_up_in[z,x],x] = 1

    # Determine the lower edge of the spongy mesophyll
    c = (img4conc2==spongy)
    d = (tileB*c)
    s_low_in = np.argmin(d, axis=1)
    # Determins the lower edge of vein
    c = (img4conc2==vein)
    d = (tileB*c)
    p_low_in = np.argmin(d,axis=1)
    # Determine the lower edge of ias
    c = (img4conc2==ias)
    d = (tileB*c)
    ias_low_in = np.argmin(d,axis=1)
    # Determine the lower edge of vein
    c = (img4conc2==vein)
    d = (tileB*c)
    v_low_in = np.argmin(d,axis=1)
    #Determine the inner edge of the lower epidermis
    c = (img4conc2==epidermis)
    d = (tileB*c)
    e_low_in = np.argmax(d, axis=1)
    e_low_in_adjust = np.array(e_low_in, copy=True)
    e_low_in_adjust[(e_low_in==hold)] = 0
    e_low_in = np.maximum(e_low_in_adjust, s_low_in)
    e_low_in = np.maximum(e_low_in_adjust, p_low_in)
    e_low_in = np.maximum(e_low_in_adjust, ias_low_in)
    e_low_in = np.maximum(e_low_in_adjust, v_low_in)

    epi_low_in = np.zeros(img.shape)
    for z in range(0,epi_low_in.shape[0]):
        for x in range(0,epi_low_in.shape[2]):
            if x > 1:
                if e_low_in[z,x]==0 or e_low_in[z,x]==hold:
                    e_low_in[z,x] = e_low_in[z,x-1]
                    epi_low_in[z,e_low_in[z,x]+hold-1,x] = 1
                else:
                    epi_low_in[z,e_up_in[z,x]+hold-1,x] = 1
            else:
                epi_low_in[z,e_low_in[z,x]+hold-1,x] = 1

    #add lower and upper halves
    epi_inner_trace = np.add(epi_low_in,epi_up_in)
    # Generate a binary stack with the pixels inside the inner epidermis trace set equal to 1
    epi_inner_up = np.zeros(img4conc1.shape, dtype=np.uint16)
    for y in tqdm(range(0,epi_inner_up.shape[2])):
        for z in range(0,epi_inner_up.shape[0]):
            epi_inner_up[z,:e_up_in[z,y],y] = 1

    epi_inner_down = np.zeros(img4conc2.shape, dtype=np.uint16)
    for y in tqdm(range(0,epi_inner_down.shape[2])):
        for z in range(0,epi_inner_down.shape[0]):
            epi_inner_down[z,:e_low_in[z,y],y] = 1
    epi_inner_down = (epi_inner_down==0)*1
    # Concatenate two halves of image
    epi_inner_fill = np.concatenate((epi_inner_up,epi_inner_down), axis = 1)
    epi_inner_fill_invert = (epi_inner_fill==0)*1
    # Set all background identified as IAS that lies outside epidermal boundaries as BG
    # Set all IAS identified as BG that lies within epidermal boundaries as IAS
    img5 = np.array(img4, copy=True)
    img5[(img4==ias)*(epi_inner_fill==1)] = bg
    img5[(img4==bg)*(epi_inner_fill_invert==1)] = ias

    return img5

def delete_dangling_epidermis(img,epidermis,background):
    # Remove 'dangling' epidermal pixels
    epid = (img==epidermis)
    epid_rmv_parts = np.array(epid, copy=True)
    for i in tqdm(range(0,epid_rmv_parts.shape[0])):
        epid_rmv_parts[i,:,:] = remove_small_objects(epid[i,:,:], min_size=800)
    # Write an array of just the removed particles
    epid_parts = epid ^ epid_rmv_parts
    # Replace the small connected epidermal particles (< 800 px^2) with BG value
    img[epid_parts==1] = background
    # Do this again in another dimension
    epid2 = (epid_rmv_parts==1)
    epid_rmv_parts2 = np.array(epid2, copy=True)
    for j in range(0,epid_rmv_parts.shape[1]):
        epid_rmv_parts2[:,j,:] = remove_small_objects(epid2[:,j,:], min_size=200)
    # Write an array of just the removed particles, again
    epid_parts2 = epid ^ epid_rmv_parts2
    # Replace the small connected epidermal particles (< 800 px^2) with BG value
    img[epid_parts2==1] = background
    # Free up some memory
    del epid_rmv_parts
    del epid_rmv_parts2
    del epid
    return img

def dbl_pct_filt(arr):
    # Define percentile filter for clipping off artefactual IAS protrusions due to dangling epidermis
    out = percentile_filter(percentile_filter(arr,size=30,percentile=10),size=30,percentile=90)
    return out

def min_max_filt(arr):
    # Define minimmum and maximum filters for clipping off artefactual IAS protrusions due to dangling epidermis
    # FIX: Perhaps make this variable? User input based?
    out = minimum_filter(maximum_filter(arr,20),20)
    return out

def check_array_orient(arr1,arr2):
    global arr1_obs
    if arr1.shape[1] != arr2.shape[1] and arr1.shape[2] != arr2.shape[2]:
        if arr1.shape[0] != arr2.shape[0]:
            if arr1.shape[0] == arr2.shape[1]:
                if arr1.shape[1] == arr2.shape[0]:
                    arr1_obs = [1,0,2]
                else:
                    arr1_obs = [1,2,0]
            else:
                if arr1.shape[1] == arr2.shape[0]:
                    arr1_obs = [2,0,1]
                else:
                    arr1_obs = [2,1,0]
        else:
            if arr1.shape[2] == arr2.shape[1]:
                arr1_obs = [0,2,1]
            else:
                arr1_obs = [0,1,2]
        out = np.moveaxis(arr2, source=arr1_obs, destination=[0,1,2])
    else:
        out = np.copy(arr2)
    return out

def winVar(img, wlen):
    # Variance filter
    wmean, wsqrmean = (cv2.boxFilter(x,-1,(wlen,wlen), borderType=cv2.BORDER_REFLECT)
                       for x in (img, img*img))
    return wsqrmean - wmean*wmean

def RFPredictCTStack(rf_transverse,gridimg_in, phaseimg_in, localthick_cellvein_in, section):
    # Use random forest model to predict entire CT stack on a slice-by-slice basis
    global dist_edge_FL
    dist_edge_FL = []
    # Define distance from lower/upper image boundary
    dist_edge = np.ones(gridimg_in.shape, dtype=np.float64)
    dist_edge[:,(0,1,2,3,4,gridimg_in.shape[1]-4,gridimg_in.shape[1]-3,gridimg_in.shape[1]-2,gridimg_in.shape[1]-1),:] = 0
    #dist_edge = transform.rescale(dist_edge, 0.25)
    dist_edge_FL = spim.distance_transform_edt(dist_edge)
    #dist_edge_FL = np.multiply(transform.rescale(dist_edge_FL,4),4)
    if dist_edge_FL.shape[1]>gridimg_in.shape[1]:
        dist_edge_FL = dist_edge_FL[:,0:gridimg_in.shape[1],:]
    # Define numpy array for storing class predictions
    RFPredictCTStack_out = np.empty(gridimg_in.shape, dtype=np.float64)
    # Define empty numpy array for feature layers (FL)
    FL = np.empty((gridimg_in.shape[1],gridimg_in.shape[2],num_feature_layers), dtype=np.float64)
    for j in tqdm(range(0,gridimg_in.shape[0])):
        # Populate FL array with feature layers using custom filters, etc.
        FL[:,:,0] = gridimg_in[j,:,:]
        FL[:,:,1] = phaseimg_in[j,:,:]
        FL[:,:,2] = gaussian(FL[:,:,0],8)
        FL[:,:,3] = gaussian(FL[:,:,1],8)
        FL[:,:,4] = gaussian(FL[:,:,0],64)
        FL[:,:,5] = gaussian(FL[:,:,1],64)
        FL[:,:,6] = winVar(FL[:,:,0],9)
        FL[:,:,7] = winVar(FL[:,:,1],9)
        FL[:,:,8] = winVar(FL[:,:,0],18)
        FL[:,:,9] = winVar(FL[:,:,1],18)
        FL[:,:,10] = winVar(FL[:,:,0],36)
        FL[:,:,11] = winVar(FL[:,:,1],36)
        FL[:,:,12] = winVar(FL[:,:,0],72)
        FL[:,:,13] = winVar(FL[:,:,1],72)
        FL[:,:,14] = LoadCTStack(localthick_cellvein_in,j,section)[:,:]
        FL[:,:,15] = dist_edge_FL[j,:,:]
        FL[:,:,16] = gaussian(FL[:,:,0],4)
        FL[:,:,17] = gaussian(FL[:,:,1],4)
        FL[:,:,18] = gaussian(FL[:,:,0],32)
        FL[:,:,19] = gaussian(FL[:,:,1],32)
        FL[:,:,20] = sobel(FL[:,:,0])
        FL[:,:,21] = sobel(FL[:,:,1])
        FL[:,:,22] = gaussian(FL[:,:,20],8)
        FL[:,:,23] = gaussian(FL[:,:,21],8)
        FL[:,:,24] = gaussian(FL[:,:,20],32)
        FL[:,:,25] = gaussian(FL[:,:,21],32)
        FL[:,:,26] = gaussian(FL[:,:,20],64)
        FL[:,:,27] = gaussian(FL[:,:,21],64)
        FL[:,:,28] = gaussian(FL[:,:,20],128)
        FL[:,:,29] = gaussian(FL[:,:,21],128)
        FL[:,:,30] = winVar(FL[:,:,20],32)
        FL[:,:,31] = winVar(FL[:,:,21],32)
        FL[:,:,32] = winVar(FL[:,:,20],64)
        FL[:,:,33] = winVar(FL[:,:,21],64)
        FL[:,:,34] = winVar(FL[:,:,20],128)
        FL[:,:,35] = winVar(FL[:,:,21],128)
        # Collapse training data to two dimensions
        FL_reshape = FL.reshape((-1,FL.shape[2]), order="F")
        class_prediction_transverse = rf_transverse.predict(FL_reshape)
        RFPredictCTStack_out[j,:,:] = class_prediction_transverse.reshape((
            gridimg_in.shape[1],
            gridimg_in.shape[2]),
            order="F")
    return(RFPredictCTStack_out)

def check_images(prediction_prob_imgs,prediction_imgs,observed_imgs,FL_imgs,phaserec_stack,folder_name):
    # Plot images of class probabilities, predicted classes, observed classes, and feature layer of interest
    #SUPPRESS
    if os.path.exists('../results/'+folder_name+'/qc') == False:
        os.mkdir('../results/'+folder_name+'/qc')
    for i in range(0,prediction_imgs.shape[0]):
        # img1 = Image.open(prediction_prob_imgs[i,:,:,1], cmap="RdYlBu")
        location = '../results/'+folder_name+'/qc/predprobIMG'+str(i)+'.tif'
        img1 = img_as_ubyte(prediction_prob_imgs[i,:,:,1])
        io.imsave(location, img1)

        location = '../results/'+folder_name+'/qc/observeIMG'+str(i)+'.tif'
        img2 = (img_as_ubyte(observed_imgs[i,:,:].astype(np.uint64)))*85 #multiply by 85 to get values (in range 0-3) into 8-bit (0-255) distribution
        io.imsave(location, img2)

        location = '../results/'+folder_name+'/qc/predIMG'+str(i)+'.tif'
        img3 = (img_as_ubyte(prediction_imgs[i,:,:].astype(np.uint64)))*85
        io.imsave(location, img3)

        location = '../results/'+folder_name+'/qc/phaserec_stackIMG'+str(i)+'.tif'
        img4 = (img_as_ubyte(phaserec_stack[260,:,:].astype(np.uint64)))*85
        io.imsave(location, img4)

        location = '../results/'+folder_name+'/qc/feature_layerIMG'+str(i)+'.tif'
        img5 = (img_as_ubyte(FL_imgs[0,:,:,26].astype(np.uint64)))*85
        io.imsave(location, img5)
    print("\nSee 'results/yourfoldername/qc' folder for quality control images\n")

def reshape_arrays(class_prediction_prob,class_prediction,Label_test,FL_test,label_stack):
    # Reshape arrays for plotting images of class probabilities, predicted classes, observed classes, and feature layer of interest
    prediction_prob_imgs = class_prediction_prob.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2],
        class_prediction_prob.shape[1]),
        order="F")
    prediction_imgs = class_prediction.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2]),
        order="F")
    observed_imgs = Label_test.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2]),
        order="F")
    FL_imgs = FL_test.reshape((
        -1,
        label_stack.shape[1],
        label_stack.shape[2],
        num_feature_layers),
        order="F")
    return prediction_prob_imgs,prediction_imgs,observed_imgs,FL_imgs

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

def predict_testset(rf_t,FL_test):
    # predict single slices from dataset
    print("***GENERATING PREDICTED STACK***")
    class_prediction = rf_t.predict(FL_test)
    class_prediction_prob = rf_t.predict_proba(FL_test)
    return class_prediction, class_prediction_prob

def print_feature_layers(rf_t,folder_name):
    # Print feature layer importance
    file = open('../results/'+folder_name+'/FeatureLayer.txt','w')
    file.write('Our OOB prediction of accuracy for is: {oob}%'.format(oob=rf_t.oob_score_ * 100)+'\n')
    feature_layers = range(0,len(rf_t.feature_importances_))
    for fl, imp in zip(feature_layers, rf_t.feature_importances_):
        file.write('Feature_layer {fl} importance: {imp}'.format(fl=fl, imp=imp)+'\n')
    file.close()

def displayImages_displayDims(gr_s,pr_s,ls,lt_s,gp_train,gp_test,label_train,label_test):
    # FIX: print images to qc
    # for i in [label_test,label_train]:
    #     imgA = ls[i,:,:]
    #     imgA = Image.fromarray(imgA)
    #     imgA.show()
    #
    # for i in [gp_train,gp_test]:
    #     io.imshow(gr_s[i,:,:], cmap='gray')
    #     io.show()
    # for i in [gp_train,gp_test]:
    #     io.imshow(pr_s[i,:,:], cmap='gray')
    #     io.show()
    # for i in [gp_train,gp_test]:
    #     io.imshow(lt_s[i,:,:])
    #     io.show()
    #check shapes of stacks to ensure they match
    print('Gridrec stack shape = ' + str(gr_s.shape))
    print('Phaserec stack shape = ' + str(pr_s.shape))
    print('Label stack shape = ' + str(ls.shape))
    print('Local thickness stack shape = ' + str(lt_s.shape))

def LoadCTStack(gridimg_in,sub_slices,section):
    # Define image dimensions
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    # Load training label data
    labelimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
    labelimg_in_rot_sub = labelimg_in_rot[sub_slices,:,:]
    return(labelimg_in_rot_sub)

def minFilter(img):
    filtered = sp.ndimage.filters.minimum_filter(img, size = (3,1,1))
    return filtered

def GenerateFL2(gridimg_in,phaseimg_in,localthick_cellvein_in,sub_slices,section):
    # Generate feature layers based on grid/phase stacks and local thickness stack
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    #match array dimensions again - COMMENTED OUT BECAUSE THE DIMENSIONS HAVE BEEN MATCHED BY TRIMMING
    #gridimg_in, phaseimg_in = match_array_dim(gridimg_in,phaseimg_in)
    # Rotate stacks to correct section view and select subset of slices
    if(section=="transverse"):
        gridimg_in_rot_sub = gridimg_in[sub_slices,:,:]
        phaseimg_in_rot_sub = phaseimg_in[sub_slices,:,:]
    else:
        gridimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
        phaseimg_in_rot = np.rot90(phaseimg_in, k=num_rot, axes=(rot_i,rot_j))
        gridimg_in_rot_sub = gridimg_in_rot[sub_slices,:,:]
        phaseimg_in_rot_sub = phaseimg_in_rot[sub_slices,:,:]
    # Define distance from lower/upper image boundary
    dist_edge = np.ones(gridimg_in.shape, dtype=np.float64)
    dist_edge[:, (0,1,2,3,4,gridimg_in.shape[1]-5,gridimg_in.shape[1]-4,gridimg_in.shape[1]-3,gridimg_in.shape[1]-2,gridimg_in.shape[1]-1), :] = 0
    #dist_edge = transform.rescale(dist_edge, 0.25,clip=True,preserve_range=True)
    dist_edge_FL = spim.distance_transform_edt(dist_edge)
    #dist_edge_FL = np.multiply(transform.rescale(dist_edge_FL,4,clip=True,preserve_range=True),4)
    if dist_edge_FL.shape[1]>gridimg_in.shape[1]:
        dist_edge_FL = dist_edge_FL[:,0:gridimg_in.shape[1],:]
    # Define empty numpy array for feature layers (FL)
    FL = np.empty((len(sub_slices),img_dim1,img_dim2,num_feature_layers), dtype=np.float64)
    # Populate FL array with feature layers using custom filters, etc.
    for i in tqdm(range(0,len(sub_slices))):
        FL[i,:,:,0] = gridimg_in_rot_sub[i,:,:]
        FL[i,:,:,1] = phaseimg_in_rot_sub[i,:,:]
        FL[i,:,:,2] = gaussian(FL[i,:,:,0],8)
        FL[i,:,:,3] = gaussian(FL[i,:,:,1],8)
        FL[i,:,:,4] = gaussian(FL[i,:,:,0],64)
        FL[i,:,:,5] = gaussian(FL[i,:,:,1],64)
        FL[i,:,:,6] = winVar(FL[i,:,:,0],9)
        FL[i,:,:,7] = winVar(FL[i,:,:,1],9)
        FL[i,:,:,8] = winVar(FL[i,:,:,0],18)
        FL[i,:,:,9] = winVar(FL[i,:,:,1],18)
        FL[i,:,:,10] = winVar(FL[i,:,:,0],36)
        FL[i,:,:,11] = winVar(FL[i,:,:,1],36)
        FL[i,:,:,12] = winVar(FL[i,:,:,0],72)
        FL[i,:,:,13] = winVar(FL[i,:,:,1],72)
        FL[i,:,:,14] = LoadCTStack(localthick_cellvein_in, sub_slices, section)[i,:,:] # > 5%
        FL[i,:,:,15] = dist_edge_FL[i,:,:]
        FL[i,:,:,16] = gaussian(FL[i,:,:,0],4)
        FL[i,:,:,17] = gaussian(FL[i,:,:,1],4)
        FL[i,:,:,18] = gaussian(FL[i,:,:,0],32)
        FL[i,:,:,19] = gaussian(FL[i,:,:,1],32)
        FL[i,:,:,20] = sobel(FL[i,:,:,0])
        FL[i,:,:,21] = sobel(FL[i,:,:,1])
        FL[i,:,:,22] = gaussian(FL[i,:,:,20],8)
        FL[i,:,:,23] = gaussian(FL[i,:,:,21],8)
        FL[i,:,:,24] = gaussian(FL[i,:,:,20],32)
        FL[i,:,:,25] = gaussian(FL[i,:,:,21],32)
        FL[i,:,:,26] = gaussian(FL[i,:,:,20],64)
        FL[i,:,:,27] = gaussian(FL[i,:,:,21],64)
        FL[i,:,:,28] = gaussian(FL[i,:,:,20],128)
        FL[i,:,:,29] = gaussian(FL[i,:,:,21],128)
        FL[i,:,:,30] = winVar(FL[i,:,:,20],32)
        FL[i,:,:,31] = winVar(FL[i,:,:,21],32)
        FL[i,:,:,32] = winVar(FL[i,:,:,20],64)
        FL[i,:,:,33] = winVar(FL[i,:,:,21],64)
        FL[i,:,:,34] = winVar(FL[i,:,:,20],128)
        FL[i,:,:,35] = winVar(FL[i,:,:,21],128)
    FL[:,:,:,36] = minFilter(FL[:,:,:,14])
    # Collapse training data to two dimensions
    FL_reshape = FL.reshape((-1,FL.shape[3]), order="F")
    return FL_reshape

def LoadLabelData(gridimg_in,sub_slices,section):
    # Load labeled data stack
    # Define image dimensions
    if(section=="transverse"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[0]
        rot_i = 1
        rot_j = 2
        num_rot = 0
    if(section=="paradermal"):
        img_dim1 = gridimg_in.shape[1]
        img_dim2 = gridimg_in.shape[0]
        num_slices = gridimg_in.shape[2]
        rot_i = 0
        rot_j = 2
        num_rot = 1
    if(section=="longitudinal"):
        img_dim1 = gridimg_in.shape[0]
        img_dim2 = gridimg_in.shape[2]
        num_slices = gridimg_in.shape[1]
        rot_i = 1
        rot_j = 0
        num_rot = 1
    # Load training label data
    labelimg_in_rot = np.rot90(gridimg_in, k=num_rot, axes=(rot_i,rot_j))
    labelimg_in_rot_sub = labelimg_in_rot[sub_slices,:,:]
    # Collapse label data to a single dimension
    img_label_reshape = labelimg_in_rot_sub.ravel(order="F")
    # Encode labels as categorical variable
    img_label_reshape = labenc.fit_transform(img_label_reshape)
    return(img_label_reshape)

def load_trainmodel(folder_name):
    print("***LOADING TRAINED MODEL***")
    #load the model from disk
    filename = folder_name+'/RF_model.sav'
    rf = pickle.load(open(filename, 'rb'))
#    print("***LOADING FEATURE LAYER ARRAYS***")
#    FL_tr = io.imread('../results/'+folder_name+'/FL_train.tif')
#    FL_te = io.imread('../results/'+folder_name+'/FL_test.tif')
#    print("***LOADING LABEL IMAGE VECTORS***")
#    Label_tr = io.imread('../results/'+folder_name+'/Label_train.tif')
#    Label_te = io.imread('../results/'+folder_name+'/Label_test.tif')
    return rf#,FL_tr,FL_te,Label_tr,Label_te

def save_trainmodel(rf_t,folder_name):#,FL_train,FL_test,Label_train,Label_test,folder_name):
    #Save model to disk; This can be a pretty large file -- >2 Gb
    print("***SAVING TRAINED MODEL***")
    filename = folder_name+'/RF_model.sav'
    pickle.dump(rf_t, open(filename, 'wb'))
#    print("***SAVING FEATURE LAYER ARRAYS***")
#    #save training and testing feature layer array
#    #SUPPRESS
#    io.imsave(folder_name+'/FL_train.tif',img_as_int(FL_train/65535))
#    io.imsave(folder_name+'/FL_test.tif',img_as_int(FL_test/65535))
#    print("***SAVING LABEL IMAGE VECTORS***")
#    #save label image vectors
#    #SUPPRESS
#    io.imsave(folder_name+'/Label_train.tif',img_as_ubyte(Label_train))
#    io.imsave(folder_name+'/Label_test.tif',img_as_ubyte(Label_test))

def train_model(gr_s,pr_s,ls,lt_s,gp_train,gp_test,label_train,label_test):
    print("***GENERATING FEATURE LAYERS***")
    #generate training and testing feature layer array
    FL_train_transverse = GenerateFL2(gr_s, pr_s, lt_s, gp_train, "transverse")
    FL_test_transverse = GenerateFL2(gr_s, pr_s, lt_s, gp_test, "transverse")
    print("***LOAD AND ENCODE LABEL IMAGE VECTORS***")
    # Load and encode label image vectors
    Label_train = LoadLabelData(ls, label_train, "transverse")
    Label_test = LoadLabelData(ls, label_test, "transverse")
    print("***TRAINING MODEL***\n(this step may take a few minutes...)")
    # Define Random Forest classifier parameters and fit model
    rf_trans = RandomForestClassifier(n_estimators=50, verbose=True, oob_score=True, n_jobs=-1, warm_start=False) #, class_weight="balanced")
    rf_trans = rf_trans.fit(FL_train_transverse, Label_train)
    return rf_trans#,FL_train_transverse,FL_test_transverse, Label_train, Label_test

def match_array_dim_label(stack1,stack2):
    #distinct match array dimensions function, to account for label_stack.shape[0]
    if stack1.shape[1]>stack2.shape[1]:
        stack1 = stack1[:,0:stack2.shape[1],:]
    else:
        stack2 = stack2[:,0:stack1.shape[1],:]
    if stack1.shape[2]>stack2.shape[2]:
        stack1 = stack1[:,:,0:stack2.shape[2]]
    else:
        stack2 = stack2[:,:,0:stack1.shape[2]]
    return stack1, stack2

def match_array_dim(stack1,stack2):
    # Match array dimensions
    if stack1.shape[0] > stack2.shape[0]:
        stack1 = stack1[0:stack2.shape[0],:,:]
    else:
        stack2 = stack2[0:stack1.shape[0],:,:]
    if stack1.shape[1] > stack2.shape[1]:
        stack1 = stack1[:,0:stack2.shape[1],:]
    else:
        stack2 = stack2[:,0:stack1.shape[1],:]
    if stack1.shape[2] > stack2.shape[2]:
        stack1 = stack1[:,:,0:stack2.shape[2]]
    else:
        stack2 = stack2[:,:,0:stack1.shape[2]]
    return stack1, stack2

def local_thickness(im):
    # Calculate local thickness; from Porespy library
    if im.ndim == 2:
        from skimage.morphology import square
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt, decimals=0))
    # Below absolutely needs float64 to work!
    im_new = sp.zeros_like(im)
    for r in tqdm(sizes):
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
        #Trim outer edge of features to remove noise
    if im.ndim == 3:
        im_new = spim.binary_erosion(input=im, structure=ball(1))*im_new
    if im.ndim == 2:
        im_new = spim.binary_erosion(input=im, structure=disc(1))*im_new
    return im_new



def localthick_up_save(folder_name, sample_name, keep_in_memory=False):
    # run local thickness, upsample and save as a .tif stack in images folder
    print("***GENERATING LOCAL THICKNESS STACK***")
    #load thresholded binary downsampled images for local thickness
    GridPhase_invert_ds = io.imread(folder_name+sample_name+'GridPhase_invert_ds.tif')
    #run local thickness
    local_thick = local_thickness(GridPhase_invert_ds)
    #local_thick_upscale = transform.rescale(local_thick, 4, mode='reflect')
    print("***SAVING LOCAL THICKNESS STACK***")
    #write as a .tif file in our images folder
    io.imsave(folder_name+sample_name+'local_thick.tif', local_thick)
    if keep_in_memory == True:
        return local_thick
    #Can be saved as ubyte as it is only integers and I doubt there will be values larger than 256
    #io.imsave(folder_name+'/local_thick_int.tif', img_as_int(local_thick/65536))

# Written by GTR
# Let'S see if this save some memory
def localthick_load_and_resize(folder_name, sample_name, threshold_rescale_factor):
    localthick_small = io.imread(folder_name+sample_name+'local_thick.tif')
    if threshold_rescale_factor > 1:
        localthick_stack = transform.resize(localthick_small, [localthick_small.shape[0]*threshold_rescale_factor,localthick_small.shape[1],localthick_small.shape[2]], order=0)
    else:
        localthick_stack = localthick_small
    return localthick_stack



# GTR: Added a saving switch so to not write it to disk if needed.
def Threshold_GridPhase_invert_down(grid_img, phase_img, Th_grid, Th_phase,folder_name,sample_name,rescale_factor):
    # Threshold grid and phase images and add the IAS together, invert, downsample and save as .tif stack
    print("***THRESHOLDING IMAGES***")
    tmp = np.zeros(grid_img.shape, dtype=np.bool_)
    tmp[grid_img < Th_grid] = 1
    tmp[grid_img >= Th_grid] = 0
    tmp[phase_img < Th_phase] = 1
    #invert
    tmp_invert = invert(tmp)
    #downsample to 50%
    #Use resize instead of rescale to keep z dimension the same
    #tmp_invert_ds = transform.resize(tmp_invert, (tmp_invert.shape[0]/rescale_factor,tmp_invert.shape[1]/rescale_factor,tmp_invert.shape[2]/rescale_factor))
    if rescale_factor == 1:
        print("***SAVING IMAGE STACK***")
        io.imsave(folder_name+'/'+sample_name+'GridPhase_invert_ds.tif',img_as_ubyte(tmp_invert))
    else:
        tmp_invert_ds = transform.resize(tmp_invert, [tmp_invert.shape[0]/rescale_factor, tmp_invert.shape[1], tmp_invert.shape[2]], order=0)
        print("***SAVING IMAGE STACK***")
        io.imsave(folder_name+'/'+sample_name+'GridPhase_invert_ds.tif',img_as_ubyte(tmp_invert_ds))



def openAndReadFile(filename):
    #opens and reads '.txt' file made by user with instructions for program...may execute full process n times
    #initialize empty lists
    gridphase_train_slices_subset = []
    gridphase_test_slices_subset = []
    label_train_slices_subset = []
    label_test_slices_subset = []
    #opens file
    myFile = open(filename, "r")
	#reads a line
    filepath = str(myFile.readline()) #function to read ONE line at a time
    filepath = filepath.replace('\n','') #strip linebreaks
    grid_name = str(myFile.readline())
    grid_name = grid_name.replace('\n','')
    phase_name = str(myFile.readline())
    phase_name = phase_name.replace('\n','')
    label_name = str(myFile.readline())
    label_name = label_name.replace('\n','')
    Th_grid = float(myFile.readline())
    Th_phase = float(myFile.readline())
    line = myFile.readline().split(",")
    for i in line:
        i = int(i.rstrip('\n'))
        gridphase_train_slices_subset.append(i)
    line = myFile.readline().split(",")
    for i in line:
        i = int(i.rstrip('\n'))
        gridphase_test_slices_subset.append(i)
    line = myFile.readline().split(",")
    for i in line:
        i = int(i.rstrip('\n'))
        label_train_slices_subset.append(i)
    line = myFile.readline().split(",")
    for i in line:
        i = int(i.rstrip('\n'))
        label_test_slices_subset.append(i)
    image_process_bool = str(myFile.readline().rstrip('\n'))
    train_model_bool = str(myFile.readline().rstrip('\n'))
    full_stack_bool = str(myFile.readline().rstrip('\n'))
    post_process_bool = str(myFile.readline().rstrip('\n'))
    epid_value = int(myFile.readline().rstrip('\n'))
    bg_value = int(myFile.readline().rstrip('\n'))
    spongy_value = int(myFile.readline().rstrip('\n'))
    palisade_value = int(myFile.readline().rstrip('\n'))
    ias_value = int(myFile.readline().rstrip('\n'))
    vein_value = int(myFile.readline().rstrip('\n'))
    folder_name = str(myFile.readline()) #function to read ONE line at a time
    #closes the file
    myFile.close()
    return filepath,grid_name,phase_name,label_name,Th_grid,Th_phase,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset,image_process_bool,train_model_bool,full_stack_bool,post_process_bool,epid_value,bg_value,spongy_value,palisade_value,ias_value,vein_value,folder_name



# This is to get the number of lines (i.e. pixels) to remove on each dimension
# to get a stack that can be resized by the defined rescaling factor
# Written by GTR
def Trim_Individual_Stack(stack, rescale_factor):
    print("***trimming stack***")
    shape_array = np.array(stack.shape) - np.array([np.repeat(0,3), np.repeat(1,3), np.repeat(2,3), np.repeat(3,3), np.repeat(4,3), np.repeat(5,3)])
    dividers_mat = shape_array % rescale_factor
    to_trim = np.argmax(dividers_mat == 0, axis=0)
    for i in np.arange(len(to_trim)):
        if to_trim[i] == 0:
            pass
        else:
            to_delete = np.arange(stack.shape[i]-to_trim[i], stack.shape[i])
            stack = np.delete(stack, to_delete, axis=i)
    return stack, to_trim



# Written by GTR
def Trimming_Stacks(filepath, gridrec_stack, phaserec_stack, label_stack, rescale_factor, grid_name,phase_name,label_name):
    gridrec_stack = Trim_Individual_Stack(gridrec_stack, rescale_factor)
    phaserec_stack = Trim_Individual_Stack(phaserec_stack, rescale_factor)
    shape_array = np.array(label_stack.shape) - np.array([np.repeat(0,3), np.repeat(1,3), np.repeat(2,3), np.repeat(3,3), np.repeat(4,3), np.repeat(5,3)])
    dividers_mat = shape_array % rescale_factor
    to_trim = np.argmax(dividers_mat == 0, axis=0)
    for i in np.arange(len(to_trim)):
        if i == 0:
            pass
        else:
            if to_trim[i] == 0:
                pass
            else:
                to_delete = np.arange(label_stack.shape[i]-to_trim[i], label_stack.shape[i])
                label_stack = np.delete(label_stack, to_delete, axis=i)
    if np.any(to_trim != 0):
        print("***SAVING TRIMMED LABELLED STACK***")
        io.imsave(filepath + label_name, img_as_ubyte(label_stack))
        print("***SAVING TRIMMED GRID STACK***")
        io.imsave(filepath + grid_name, img_as_ubyte(gridrec_stack))
        print("***SAVING TRIMMED PHASE STACK***")
        io.imsave(filepath + phase_name, img_as_ubyte(phaserec_stack))
    return gridrec_stack, phaserec_stack, label_stack, to_trim



# Written by GTR
def Load_Individual_images(fp,name,rescale_factor):
    print("***LOADING INDIVIDUAL IMAGE STACK***")
    # Read gridrec, phaserec, and label tif stacks
    stack = io.imread(fp + name)
    stack = Trim_Individual_Stack(stack, rescale_factor)
    if rescale_factor > 1:
        stack = Resize_and_save_stack(stack, name, rescale_factor, fp, keep_in_memory=True)
    return stack



def Load_images(fp,gr_name,pr_name,ls_name):
    print("***LOADING IMAGE STACKS***")
    # Read gridrec, phaserec, and label tif stacks
    gridrec_stack = io.imread(fp + gr_name)
    phaserec_stack = io.imread(fp + pr_name)
    label_stack = io.imread(fp + ls_name)
    #FIX: Invert my label_stack, uncomment as needed
    label_stack = invert(label_stack)
    # Reorient label stack
    label_stack = check_array_orient(gridrec_stack,label_stack)
    return gridrec_stack, phaserec_stack, label_stack



def load_fullstack(filename,folder_name):
    # print("***LOADING FULL STACK PREDICTION***")
    #load the model from disk
    rf = io.imread('../results/'+folder_name+'/'+filename)
    return rf


def displayPixelvalues(stack):
    pixelVals = np.unique(stack)
    for i in range(0,len(pixelVals)):
        print('Class '+str(i)+' has a pixel value of: '+str(pixelVals[i]))


# Written by GTR
# To resize stacks in 2 dimensions like in ImageJ
# This loads the original image, trims the edges so that it can be divided by the rescale_factor,
# then resizes each slice in the x and y axis (so keep the same number of slices).
# In python notation, this ends up as [z, x/2, y/2]
def Load_Resize_and_Save_Stack(filepath, stack_name, rescale_factor, keep_in_memory=True, labelled_stack=False):
    if os.path.isfile(filepath + stack_name + "_" + str(rescale_factor) +"x-smaller.tif"):
        stack_rs = io.imread(filepath + stack_name + "_" + str(rescale_factor) +"x-smaller.tif")
        return stack_rs
    else:
        stack = io.imread(filepath + stack_name)
        if labelled_stack==False:
            stack, to_trim = Trim_Individual_Stack(stack, rescale_factor)
            print(to_trim)
            if np.any(to_trim):
                print("***SAVING TRIMMED STACK " + stack_name + "***")
                io.imsave(filepath + stack_name, img_as_ubyte(stack), imagej=True)
        print("***RESIZING***")
        #Iterating over each slice is faster than doing it in one call with transform.resize
        stack_rs = np.empty(np.array(stack.shape)/np.array([1,rescale_factor,rescale_factor]))
        for idx in np.arange(stack_rs.shape[0]):
            stack_rs[idx] = transform.resize(stack[idx], [stack.shape[1]/rescale_factor, stack.shape[2]/rescale_factor], order=0)
        print("***SAVING RESIZED STACK " + stack_name + "***")
        io.imsave(filepath + stack_name + "_" + str(rescale_factor) +"x-smaller.tif", img_as_ubyte(stack_rs))
        if keep_in_memory == True:
            return img_as_ubyte(stack_rs)
