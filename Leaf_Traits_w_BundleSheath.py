#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:03:05 2018

@author: gtrancourt
"""

#%%
import numpy as np
import os
from pandas import DataFrame
from skimage import transform, img_as_bool, img_as_int, img_as_ubyte, img_as_float32
import skimage.io as io
from skimage.measure import label, marching_cubes_lewiner, mesh_surface_area, regionprops, marching_cubes_classic
import zipfile
#import cv2

def Trim_Individual_Stack(large_stack, small_stack):
    print("***trimming stack***")
    dims = np.array(binary_stack.shape, dtype='float') / np.array(raw_pred_stack.shape, dtype='float')
    if np.all(dims <= 2):
        return large_stack
    else:
        if dims[1] > 2:
            if (large_stack.shape[1]-1)/2 == small_stack.shape[1]:
                large_stack = np.delete(large_stack, large_stack.shape[1]-1, axis=1)
            else:
                if (large_stack.shape[1]-2)/2 == small_stack.shape[1]:
                    large_stack = np.delete(large_stack, np.arange(large_stack.shape[1]-2, large_stack.shape[1]), axis=1)
        if dims[2] > 2:
            if (large_stack.shape[2]-1)/2 == small_stack.shape[2]:
                large_stack = np.delete(large_stack, large_stack.shape[2]-1, axis=2)
            else:
                if (large_stack.shape[2]-2)/2 == small_stack.shape[2]:
                    large_stack = np.delete(large_stack, np.arange(large_stack.shape[2]-2, large_stack.shape[2]), axis=2)
        return large_stack


#%%
#Pixel dimmension
px_edge = 0.636 #µm
vx_volume = px_edge**3

#%%

#Load segmented image
base_folder_name = '/run/media/gtrancourt/GTR_Touro/Grasses_uCT/'
sample_name = 'Pguttata96_1176_'
folder_name = 'MLresults_second/'
binary_filename = sample_name + 'BINARY.tif' # sample_name + '_BINARY-8bit-CROPPED.tif'
raw_ML_prediction_name = 'fullstack_prediction.tif'

filepath = base_folder_name + sample_name + '/'

#%%
# Check if the file has already been processed -- Just in case!
if os.path.isfile(filepath + sample_name + 'RESULTS.txt'):
    print('This file has already been processed!')
    assert False


#%%
# Load the ML segmented stack
raw_pred_stack = io.imread(filepath + folder_name + raw_ML_prediction_name)
print(np.unique(raw_pred_stack[100]))

# Trim at the edges -- The ML does a bad job there
# Here I remove 50 slices at the beginning and the end, 
# and 40 pixels at the left and right edges
trim_slices = 50
trim_column = 40

raw_pred_stack = raw_pred_stack[trim_slices:-trim_slices,:,trim_column:-trim_column]

io.imshow(raw_pred_stack[100])
#%%
# Define the values for each tissue

epid_value = 51
bg_value = 204
mesophyll_value = 0
ias_value = 255
vein_value = 102
bs_value = 153

#%%
###################
## EPIDERMIS
###################

# Label all of the epidermis regions
unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

io.imshow(unique_epidermis_volumes[100])
#%%
# Find the size and properties of the epidermis regions
epidermis_area = np.zeros(len(props_of_unique_epidermis))
epidermis_label = np.zeros(len(props_of_unique_epidermis))
epidermis_centroid = np.zeros([len(props_of_unique_epidermis),3])
for regions in np.arange(len(props_of_unique_epidermis)):
    epidermis_area[regions] = props_of_unique_epidermis[regions].area
    epidermis_label[regions] = props_of_unique_epidermis[regions].label
    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

# Find the two largest epidermis
ordered_epidermis = np.argsort(epidermis_area)
print('The two largest values below should be in the same order of magnitude')
print(epidermis_area[ordered_epidermis[-4:]])
print('The center of the epidermis should be more or less the same on the 1st and 3rd columns')
print(epidermis_centroid[ordered_epidermis[-4:]])

two_largest_epidermis = (unique_epidermis_volumes == ordered_epidermis[-1]+1) | (unique_epidermis_volumes == ordered_epidermis[-2]+1)

#Check if it's correct
#io.imsave(filepath + folder_name + 'test_epidermis.tif',
#          img_as_ubyte(two_largest_epidermis))
io.imshow(two_largest_epidermis[100])
#%%

# Get the values again: makes it cleaner
unique_epidermis_volumes = label(two_largest_epidermis, connectivity=1)
props_of_unique_epidermis = regionprops(unique_epidermis_volumes)
epidermis_area = np.zeros(len(props_of_unique_epidermis))
epidermis_label = np.zeros(len(props_of_unique_epidermis))
epidermis_centroid = np.zeros([len(props_of_unique_epidermis),3])
for regions in np.arange(len(props_of_unique_epidermis)):
    epidermis_area[regions] = props_of_unique_epidermis[regions].area
    epidermis_label[regions] = props_of_unique_epidermis[regions].label
    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

#io.imshow(unique_epidermis_volumes[100])

# Transform the array to 8-bit: no need for the extra precision as there are only 3 values
unique_epidermis_volumes = np.array(unique_epidermis_volumes, dtype='uint8')

# Find the fvalues of each epidermis: assumes adaxial epidermis is at the top of the image
adaxial_epidermis_value = unique_epidermis_volumes[100,:,100][(unique_epidermis_volumes[100,:,100] != 0).argmax()]
abaxial_epidermis_value = int(np.arange(start=1,stop=3)[np.arange(start=1,stop=3) != adaxial_epidermis_value])

# Compute volume
epidermis_adaxial_volume = epidermis_area[adaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)
epidermis_abaxial_volume = epidermis_area[abaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)

# Tichkness return a 2D array, i.e. the thcikness of each column
epidermis_abaxial_thickness = np.sum((unique_epidermis_volumes == abaxial_epidermis_value), axis=1) * (px_edge*2)
epidermis_adaxial_thickness = np.sum((unique_epidermis_volumes == adaxial_epidermis_value), axis=1) * (px_edge*2)

#%%
###################
## VEINS
###################

# Get the veins volumes
unique_vein_volumes = label(raw_pred_stack == vein_value, connectivity=1)
props_of_unique_veins = regionprops(unique_vein_volumes)

io.imshow(unique_vein_volumes[100])
#%%
veins_area = np.zeros(len(props_of_unique_veins))
veins_label = np.zeros(len(props_of_unique_veins))
veins_centroid = np.zeros([len(props_of_unique_veins),3])
for regions in np.arange(len(props_of_unique_veins)):
    veins_area[regions] = props_of_unique_veins[regions].area
    veins_label[regions] = props_of_unique_veins[regions].label
    veins_centroid[regions] = props_of_unique_veins[regions].centroid

# Find the largest veins
ordered_veins = np.argsort(veins_area)
#veins_area[ordered_veins[-80:]]
#veins_area[ordered_veins[:1000]]
#veins_centroid[ordered_veins[-4:]]

#print(np.sum(veins_area <= 1000))

# I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
# the noise left by the segmentation method and kept only the largest veins.
# This should be adjusted depending on the species/images/maginification.
large_veins_ids = veins_label[veins_area > 100000]

largest_veins = np.in1d(unique_vein_volumes, large_veins_ids).reshape(raw_pred_stack.shape)

# Get the values again
vein_volume = np.sum(largest_veins) * (px_edge * (px_edge*2)**2)

#Check if it's correct
#io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_veins.tif',
#          img_as_ubyte(largest_veins))
io.imshow(largest_veins[100])


#%%
###################
## BUNDLE SHEATHS
###################

# Get the veins volumes
unique_bs_volumes = label(raw_pred_stack == bs_value, connectivity=1)
props_of_unique_bs = regionprops(unique_bs_volumes)

io.imshow(unique_bs_volumes[100])
#%%
bs_area = np.zeros(len(props_of_unique_bs))
bs_label = np.zeros(len(props_of_unique_bs))
bs_centroid = np.zeros([len(props_of_unique_bs),3])
for regions in np.arange(len(props_of_unique_bs)):
    bs_area[regions] = props_of_unique_bs[regions].area
    bs_label[regions] = props_of_unique_bs[regions].label
    bs_centroid[regions] = props_of_unique_bs[regions].centroid

# Find the largest bs
ordered_bs = np.argsort(bs_area)
#bs_area[ordered_bs[-80:]]
#bs_area[ordered_bs[:1000]]
#bs_centroid[ordered_bs[-4:]]

#print(np.sum(bs_area <= 1000))

# I found that for my images, a threshold of 100000 (1e5) pixel^3 removed
# the noise left by the segmentation method and kept only the largest bs.
# This should be adjusted depending on the species/images/maginification.
large_bs_ids = bs_label[bs_area > 100000]

largest_bs = np.in1d(unique_bs_volumes, large_bs_ids).reshape(raw_pred_stack.shape)

# Get the values again
bs_volume = np.sum(largest_bs) * (px_edge * (px_edge*2)**2)

#Check if it's correct
#io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_bs.tif',
#          img_as_ubyte(largest_bs))
io.imshow(largest_bs[100])


#%%
###################
## AIRSPACE
###################

#########################################
## CREATE THE FULLSIZE SEGMENTED STACK ##
#########################################

# My segmenteation procedure used a reduced size stack, since my original
# images are too big to be handled. I do want to use my original images for
# their quality and details, so I use the binary image and add on top of it
# the background, epidermis, and veins that have been segmented. That way, I
# keep the detail I want at the airspace-cell interface, while still having a
# good background, epidermis, and vein segmentation to remove the tissues that
# are not need for some traits.

##############################
## LOADING THE BINARY STACK ##
## IN ORIGINAL SIZE         ##
##############################

# I've started compressing my files. The code below extracts the file,
# loads it into memory, and then deletes the file (it's still in memory).
# The commented code at the end loads the uncompressed image.

#Load the compressed binary stack in the original dimensions

#binary_zip = zipfile.ZipFile(base_folder_name + sample_name + '/' + binary_filename + '.zip', 'r')
#binary_zip.extractall(base_folder_name + sample_name + '/')
#binary_raw = binary_zip.open(sample_name + '/' + binary_filename)
binary_stack = img_as_bool(io.imread(filepath + '/' + binary_filename))

binary_stack = binary_stack[trim_slices:-trim_slices,:,:]
binary_stack = binary_stack[:,:,(trim_column*2):(-trim_column*2)]

#os.remove(base_folder_name + sample_name + '/' + sample_name + '/' + binary_filename)
#os.rmdir(base_folder_name + sample_name + '/' + sample_name)


io.imshow(binary_stack[100])


#%%
#Check and trim the binary stack if necessary
# This is to match the dimensions between all images
# Basically, it trims odds numbered dimension so to be able to divide/multiply them by 2.
binary_stack = Trim_Individual_Stack(binary_stack, raw_pred_stack)

# TO MANUALLY DELETE SOME SLICES
#binary_stack = np.delete(binary_stack, 910, axis=1)
binary_stack = np.delete(binary_stack, 818, axis=0)

#binary_stack = np.delete(binary_stack, np.arange(0, 160*2), axis=2)


#%%
# This cell creates an empty array filled with the backgroud color (177), then
# adds all of the leaf to it. Looping over each slice (this is more memory
# efficient than working on the whole stack), it takes the ML segmented image,
# resize the slice, and adds it to the empty array.
bg_value_new = 177
vein_value_new = 147
ias_value_new = 255
bs_value_new = 102

large_segmented_stack = np.full(shape=binary_stack.shape, fill_value=bg_value_new, dtype='uint8') # Assign an array filled with the background value 177.
for idx in np.arange(large_segmented_stack.shape[0]):
    # Creates a boolean 2D array of the veins (from the largest veins id'ed earlier)
    temp_veins = img_as_bool(transform.resize(largest_veins[idx],
                                              [binary_stack.shape[1], binary_stack.shape[2]],
                                              anti_aliasing=False, order=0))
    # Creates a boolean 2D array of the bundle sheath (from the largest veins id'ed earlier)
    temp_bs = img_as_bool(transform.resize(largest_bs[idx],
                                              [binary_stack.shape[1], binary_stack.shape[2]],
                                              anti_aliasing=False, order=0))
    # Creates a 2D array with the epidermis being assinged values 30 or 60
    temp_epid = transform.resize(unique_epidermis_volumes[idx],
                                              [binary_stack.shape[1], binary_stack.shape[2]],
                                              anti_aliasing=False, preserve_range=True, order=0) * 30
    # Creates a 2D mask of only the leaf to remove the backgroud from the
    # original sized binary image.
    leaf_mask = img_as_bool(transform.resize(raw_pred_stack[idx] != bg_value,
                                             [binary_stack.shape[1], binary_stack.shape[2]],
                                             anti_aliasing=False, order=0))
    large_segmented_stack[idx][leaf_mask] = binary_stack[idx][leaf_mask] * ias_value_new #binary_stack is a boolean, so you need to multiply it.
    large_segmented_stack[idx][temp_veins] = vein_value_new
    large_segmented_stack[idx][temp_bs] = bs_value_new
    large_segmented_stack[idx][temp_epid != 0] = temp_epid[temp_epid != 0]

io.imshow(large_segmented_stack[100])
print('### Validate the values in the stack ###')
print(np.unique(large_segmented_stack[100]))

# Special tiff saving option for ImageJ compatibility when files larger than
# 2 Gb. It's like it doesn't recognize something if you don't turn this option
# on for large files and then ImageJ or FIJI fail to load the large stack 
# (happens on my linux machine installed with openSUSE Tumbleweed).
if large_segmented_stack.nbytes >= 2e10:
    imgj_bool = True
else:
    imgj_bool = False
    
# Save the image
io.imsave(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif', large_segmented_stack, imagej=imgj_bool)

#%%
################################################
## COMPUTE TRAITS ON THE ORIGINAL SIZED STACK ##
################################################

# Load the large segmented stack to re-run the calculations if needed

#large_segmented_stack = io.imread(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif')
#
#io.imshow(large_segmented_stack[100])
#print(np.unique(large_segmented_stack[100]))

#large_segmented_stack = np.delete(large_segmented_stack, np.arange(0,500), axis=0)

#%%
# Redefine the values for the different tissues as used in the segmented image.
# The epidermis will be defined later.
bg_value = 177
spongy_value = 0
palisade_value = 0
if spongy_value == palisade_value:
    mesophyll_value = spongy_value
else:
    mesophyll_value = [spongy_value, palisade_value]
ias_value = 255
vein_value = 147
bs_value = 102

# Find the values of each epidermis: assumes adaxial epidermis is at the top of the image
adaxial_epidermis_value = large_segmented_stack[100,:,100][(large_segmented_stack[100,:,100] != bg_value).argmax()]

if adaxial_epidermis_value == 30:
    abaxial_epidermis_value = 60
else:
    if adaxial_epidermis_value == 60:
        abaxial_epidermis_value = 30

#Measure the different volumes
leaf_volume = np.sum(large_segmented_stack != bg_value) * vx_volume
mesophyll_volume = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != adaxial_epidermis_value) & (large_segmented_stack != abaxial_epidermis_value)) * vx_volume
cell_volume = np.sum(large_segmented_stack == mesophyll_value) * vx_volume
air_volume = np.sum(large_segmented_stack == ias_value) * vx_volume
epidermis_abaxial_volume = np.sum(large_segmented_stack == abaxial_epidermis_value) * vx_volume
epidermis_adaxial_volume = np.sum(large_segmented_stack == adaxial_epidermis_value) * vx_volume
vein_volume = np.sum(large_segmented_stack == vein_value) * vx_volume

print(leaf_volume)
print(cell_volume + air_volume + epidermis_abaxial_volume + epidermis_adaxial_volume + vein_volume)


#Measure the thickness of the leaf, the epidermis, and the mesophyll
leaf_thickness = np.sum(np.array(large_segmented_stack != bg_value, dtype='bool'), axis=1) * px_edge
mesophyll_thickness = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != adaxial_epidermis_value) & (large_segmented_stack != abaxial_epidermis_value), axis=1) * px_edge
epidermis_abaxial_thickness = np.sum(large_segmented_stack == abaxial_epidermis_value, axis=1) * px_edge
epidermis_adaxial_thickness = np.sum(large_segmented_stack == adaxial_epidermis_value, axis=1) * px_edge

print(np.median(leaf_thickness),leaf_thickness.mean(),leaf_thickness.std())
print(np.median(mesophyll_thickness),mesophyll_thickness.mean(),mesophyll_thickness.std())
print(np.median(epidermis_adaxial_thickness),epidermis_adaxial_thickness.mean(),epidermis_adaxial_thickness.std())
print(np.median(epidermis_abaxial_thickness),epidermis_abaxial_thickness.mean(),epidermis_abaxial_thickness.std())

#%%
# Leaf area
# I was lazy here as I assume the leaf is parallel to the border of the image.
leaf_area = large_segmented_stack.shape[0] * large_segmented_stack.shape[2] * (px_edge**2)

#Caluculate Surface Area (adapted from Matt Jenkins' code)
# This can take quite a lot of RAM
# This gives very similar results to BoneJ. BoneJ uses the Lorensen algorithm,
# which is available as use_classic=True in te marching_cubes_lewiner() function.
# However, the help page for this function warns that the Lorensen algorithm
# might result in topologically incorrect results, and as such the Lewiner
# algorithm is better (and faster too). So it is probably a better approach.
ias_vert_faces = marching_cubes_lewiner(large_segmented_stack == ias_value_new, 0, allow_degenerate=False)#, spacing=(px_edge,px_edge,px_edge))
ias_SA = mesh_surface_area(ias_vert_faces[0],ias_vert_faces[1])
true_ias_SA = ias_SA * (px_edge**2)
print('IAS surface area: '+str(true_ias_SA)+' µm**2')
print('or '+str(float(true_ias_SA/1000000))+' mm**2')
# end Matt's code adaptation



try:
    bs_volume
except NameError:
    bs_volume=0

print('Sm: '+str(true_ias_SA/leaf_area))
print('Ames/Vmes: '+str(true_ias_SA/(mesophyll_volume-vein_volume-bs_volume)))

#%%
# NOTE ON SA CODE ABOVE
# The same procedure as for epidermises and veins could be done, i.e. using the
# label() function to identify all of the un-connected airspace volumes and
# compute the surface area on each of them. That way we can get the surface
# area of the largest airspace and the connectivity term presented in Earles
# et al. (2018) (Beyond porosity - Bromeliad paper).

# This is done in the code below.

# Label all of the ias regions
unique_ias_volumes = label(large_segmented_stack == ias_value_new, connectivity=1)
props_of_unique_ias = regionprops(unique_ias_volumes)

# Find the size and properties of the epidermis regions
ias_area = np.zeros(len(props_of_unique_ias))
ias_label = np.zeros(len(props_of_unique_ias))
ias_centroid = np.zeros([len(props_of_unique_ias),3])
ias_SA = np.zeros([len(props_of_unique_ias),3])
for regions in np.arange(len(props_of_unique_ias)):
    ias_area[regions] = props_of_unique_ias[regions].area
    ias_label[regions] = props_of_unique_ias[regions].label
    ias_centroid[regions] = props_of_unique_ias[regions].centroid

# Find the two largest ias
ordered_ias = np.argsort(ias_area)
print('Check for the largest values and adjust use the necessary value to compute the largest connected IAS.')
print(ias_area[ordered_ias[-4:]])
print(ias_label[ordered_ias[-4:]])

largests_ias = ias_label[ordered_ias[-4:]]
ias_SA = np.zeros(len(largests_ias))
for i in np.arange(len(ias_SA)):
    ias_vert_faces = marching_cubes_lewiner(unique_ias_volumes == largests_ias[i], 0, spacing=(px_edge,px_edge,px_edge))
    ias_SA[i] = mesh_surface_area(ias_vert_faces[0],ias_vert_faces[1])

# Adjust depending on if you have IAS that are large but not connected such as
#  when bundle sheath extensions are present.
largest_connected_ias_SA = sum(ias_SA[-1:])
largest_connected_ias_volume = sum(ias_area[ordered_ias[-1:]]) * vx_volume

#%%

# Write the data into a data frame
data_out = {'LeafArea':leaf_area,
            'LeafThickness':leaf_thickness.mean(),
            'LeafThickness_SD':leaf_thickness.std(),
            'MesophyllThickness':mesophyll_thickness.mean(),
            'MesophyllThickness_SD':mesophyll_thickness.std(),
            'ADEpidermisThickness':epidermis_adaxial_thickness.mean(),
            'ADEpidermisThickness_SD':epidermis_adaxial_thickness.std(),
            'ABEpidermisThickness':epidermis_abaxial_thickness.mean(),
            'ABEpidermisThickness_SD':epidermis_abaxial_thickness.std(),
            'LeafVolume':leaf_volume,
            'MesophyllVolume':mesophyll_volume,
            'ADEpidermisVolume':epidermis_adaxial_volume,
            'ABEpidermisVolume':epidermis_abaxial_volume,
            'VeinVolume':vein_volume,
            'BSVolume':bs_volume,
            'VeinBSVolume':vein_volume+bs_volume,
            'CellVolume':cell_volume,
            'IASVolume':air_volume,
            'IASSurfaceArea':true_ias_SA,
            'IASLargestConnectedVolume':largest_connected_ias_volume,
            'IASLargestConnectedSA':largest_connected_ias_SA
            }
results_out = DataFrame(data_out, index={sample_name})
# Save the data to a CSV
results_out.to_csv(base_folder_name + sample_name + '/' + sample_name + 'RESULTS.txt', sep='\t', encoding='utf-8')
