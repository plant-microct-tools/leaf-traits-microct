# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:07:35 2018

@author: Guillaume Theroux-Rancourt
"""


# Computing leaf stomatal regions
# Using some methods presented in in Earles et al. (2018),
# as well as new methods by Guillaume Théroux-Rancourt

# Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR
# (2018) Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact
# Mesophyll Conductance. (http://www.plantphysiol.org/content/178/1/148)
# Plant Physiol 178: 148-162


# __Created__ on 2020-01-02
# by Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
#
# __Last edited__ on 2020-01-02
#  by Guillaume Théroux-Rancourt
#
# Image processing note:
# - The file used needs to have stomata either drawn on the segmented image
#   (as is currently used) or on a separate image of the same dimensions as the
#   segmented image (not implemented but easily done).
# -__How I have drawn the stomata__: How I did label the stomata was in ImageJ,
#    using both the grid and segmented stacks. I replaced both so to get a
#    paradermal view, and then synchronized both windows
#    (Analyze > Tools > Synchronize windows). When I saw a stoma, I labelled it
#    at or immediately below start of the IAS at that point. For label, I just
#    use an ellipsis, but any form would do. I saved that ROI into the ROI
#    manager, and proceeded to the other stomata. When I labelled all at their
#    right location, I filled all ROI with a specific color on the segmented
#    stack in paradermal view, either manually or with the following macro:
#    https://github.com/gtrancourt/imagej_macros/blob/master/macros/fillInside-macro.txt.
#    I then replaced it to the former view (i.e. in te same direction as to get
#    the paradermal view).
#
# Notes:
# -This code works and the results are comparable to what would be done using
#  the MorpholibJ plugin in ImageJ. However, the results are not identical,
#  probably due to the different implementations of the geodesic distance
#  computation.
#
# I've got help and inspiration from:
# - https://stackoverflow.com/questions/28187867/geodesic-distance-transform-in-python
# - https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image


import sys
import os
import numpy as np
from pandas import DataFrame
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, binary_fill_holes, binary_dilation
import skfmm
import skimage.io as io
from skimage import img_as_ubyte, img_as_bool
from skimage.util import invert
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from skimage.transform import resize
import time
from tqdm import tqdm
import joblib
import multiprocessing
import gc

__author__ = "Guillaume Théroux-Rancourt"
__copyright__ = ""
__credits__ = ["Guillaume Théroux-Rancourt", "J. Mason Earles"]
__license__ = "MIT"
__version__ = ""
__maintainer__ = "Guillaume Théroux-Rancourt"
__email__ = "guillaume.theroux-rancourt@boku.ac.at"
__status__ = "beta"

t_start = time.time()

# Extract data from command line input
full_script_path = sys.argv[0]

# create python-dictionary from command line inputs (ignore first)
sys.argv = sys.argv[1:]
arg_dict = dict(j.split('=') for j in sys.argv)

# define variables for later on
filenames = []  # list of arg files
req_not_def = 0  # permission variable for required definitions when using command line option

# define important variables using command line
for key, value in arg_dict.items():
    if key == 'argfiles':
        for z in value.split(','):
            z.strip()
            z = z.replace('\n', '')
            filenames.append(z)
    if key == 'path_to_argfile_folder':
        path_to_argfile_folder = str(value)
    else:
        # read in desired values for parameters
        if key == 'path_to_sample':
            path_to_sample = str(value)
        if key == 'stomata_stack_suffix':
            stomata_stack_suffix = str(value)
        if key == 'px_edge':
            px_edge = float(value)
        if key == 'seg_values':
            seg_values = value
        if key == 'nb_cores':
            nb_cores = int(value)
        if key == 'base_path':
            base_path = str(value)
        if key == 'rescale_factor':
            rescale_factor = int(value)
        if key == 'stomata_cropping':
            if value == 'True':
                stomata_cropping = True
            else:
                stomata_cropping = False
        if key == 'fix_stomata':
            if value == 'True':
                fix_stomata = True
            else:
                fix_stomata = False
        # set up default values for some optional parameters
        try:
            rescale_factor
        except NameError:
            rescale_factor = 0
        try:
            nb_cores
        except NameError:
            nb_cores = multiprocessing.cpu_count()
        try:
            seg_values
        except NameError:
            seg_values = 'default'
        try:
            fix_stomata
        except NameError:
            fix_stomata = False
        try:
            stomata_cropping
        except NameError:
            stomata_cropping = True


# TESTING
# path_to_sample = './C_D_5_Strip1_'
# rescale_factor = 2
# px_edge = 0.1625
# seg_values = 'default'
# nb_cores = 7
# base_path = '/run/media/guillaume/Elements/Vitis_Shade_Drought/_ML_DONE/'
# stomata_stack_suffix = 'STOMATA_STACK.tif'

 
# Function to resize in all 3 dimensions
# Loops over each slice: Faster and more memory efficient
# than working on the whole array at once.
def StackResize(stack, rf=rescale_factor):
    resized_shape = np.array(stack.shape)/np.array([1, rf, rf])
    stack_rs = np.empty(shape = resized_shape.astype(np.int64))
    for idx in np.arange(stack_rs.shape[0]):
        stack_rs[idx] = resize(stack[idx],
                               [stack.shape[1]/rf, stack.shape[2]/rf],
                               order=0, preserve_range=True, 
                               anti_aliasing=False)
    resized_shape_2 = np.array(stack_rs.shape)/np.array([rf, 1, 1])
    stack_rs2 = np.empty(shape = resized_shape_2.astype(np.int64))
    for idx in np.arange(stack_rs2.shape[1]):
        stack_rs2[:, idx, :] = resize(stack_rs[:, idx, :],
                                      [stack_rs.shape[0]/rf, stack_rs.shape[2]],
                                      order=0, preserve_range=True, 
                                      anti_aliasing=False)
    return stack_rs2

# Function to threshold images (i.e. binarize them to boolean)
# Can have multiple threshold values (Th_value)
# Will threshold irrespective of bit depth: just input the right value.


def Threshold(input_img, Th_value):
    tmp = np.zeros(input_img.shape, dtype=np.bool)
    if isinstance(Th_value, int):
        tmp[input_img == Th_value] = 1
    else:
        if isinstance(Th_value, float):
            tmp[input_img > 0. & input_img < 1.] = 1
        else:
            for th_val in range(len(Th_value)):
                tmp[input_img == Th_value[th_val]] = 1
    return tmp

# Erosion3DimJ does a 1-px erosion on each slice of an image, like in ImageJ
# Used to produce the outline like in ImageJ


def Erosion3DimJ(input_img):
    tmp = np.zeros(input_img.shape, dtype=np.bool)
    for i in range(input_img.shape[0]):
        tmp[i, :, :] = binary_erosion(input_img[i, :, :])
    return tmp

# Function below is modified from
# https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image


def getLargestAirspace(input_img):
    # Label all the connected airspace
    # I specify here the background value, so we know which to remove later.
    # Connectivity=2 means 8-connected (faces+edges), =1 means 4-connected (faces only)
    labeled_img = label(input_img, background=0, connectivity=1)
    labels_index = np.column_stack((np.unique(labeled_img),
                                    np.bincount(labeled_img.flat)))
    # Get the label of the largest airspace
    labels_index_sort = labels_index[:, 1].argsort()
    if labels_index_sort[-1] == 0:
        largest_airspace_label = labels_index_sort[-2]
    else:
        largest_airspace_label = labels_index_sort[-1]
    # Create a new image
    largest_airspace = (labeled_img == largest_airspace_label)
    return largest_airspace


def get_neighbours(p, exclude_p=True, shape=None):
    # Taken from:
    # https://stackoverflow.com/questions/34905274/how-to-find-the-neighbors-of-a-cell-in-an-ndarray
    ndim = len(p)
    # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T
    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)
    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]
    neighbours = p + offsets    # apply offsets to p
    # optional: exclude out-of-bounds indices
    if shape is not None:
        valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
        neighbours = neighbours[valid]
    return neighbours


def get_bad_neighbours(pos, img_stack, value1, value2, shape):
    neigh = get_neighbours(pos, shape=shape)
    bad_neigh = np.zeros(neigh.shape[0], dtype='bool')
    for j in np.arange(neigh.shape[0]):
        bad_neigh[j] = (img_stack[tuple(neigh[j])] == value1) | (img_stack[tuple(neigh[j])] == value1)
    return np.any(bad_neigh)

# Find the 3D bounding box
def bbox2_3D(img, value):
    # From https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    z = np.any(img == value, axis=(1, 2))
    c = np.any(img == value, axis=(0, 2))
    r = np.any(img == value, axis=(0, 1))
    zmin, zmax = np.where(z)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    rmin, rmax = np.where(r)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

# Set directory of functions in order to import MLmicroCTfunctions
path_to_script = '/'.join(full_script_path.split('/')[:-1]) + '/'
base_folder_name = base_path
sample_path_split = path_to_sample.split('/')

if len(sample_path_split) == 1:
    sample_name = path_to_sample
    filename = sample_name + 'SEGMENTED.tif'
    filepath = base_folder_name + sample_name + '/'
elif len(sample_path_split) == 2:
    if sample_path_split[0] == '.':
        sample_name = sample_path_split[-1]
        if os.path.isfile(base_folder_name + sample_name + '/' + sample_name + 'SEGMENTED_w_STOMATA.tif'):
            filename = sample_name + 'SEGMENTED_w_STOMATA.tif'
        else:
            filename = sample_name + 'SEGMENTED.tif'
        filepath = base_folder_name + sample_name + '/'
    else:
        sample_name = sample_path_split[-2]
        filename = sample_path_split[-1]
        filepath = base_folder_name + sample_name + '/'
elif len(sample_path_split) == 3:
    if sample_path_split[0] == '.':
        sample_name = sample_path_split[-2]
        filename = sample_path_split[-1]
        filepath = base_folder_name + sample_name + '/'
    else:
        sample_name = sample_path_split[-3]
        base_folder_name = base_path + '/' + sample_name + '/' + sample_path_split[-2] + '/'
        filepath = base_folder_name
        filename = sample_path_split[-1]


# Create folder to store results
if not os.path.exists(filepath + 'STOMATA_and_TORTUOSITY/'):
    os.makedirs(filepath + 'STOMATA_and_TORTUOSITY/')

# Define pixel dimension to rescale factor
px_edge_rescaled = px_edge * rescale_factor

# Read composite stack including labelling of stomata
print('')
print('************************************************')
print('***STARTING STOMATAL REGIONS COMPUTATION FOR***')
print('            ' + sample_name)
print('')
print('   Base folder path: ', base_folder_name)
print('   Filepath: ', filepath)
print('   Filename: ', filename)
print('   Try to fix stomata labels: ', fix_stomata)
print('   Stomata Stack provided:',"stomata_stack_suffix" in locals())

# Check if file has already been processed
if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SINGLE-STOMA-RESULTS.txt'):
    raise ValueError('This file has already been processed!')
if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'NO_SINGLE_STOMA_REGIONS.txt'):
    raise ValueError('This file has already been processed!')

print("***LOADING AND RESIZING STACK***")
composite_stack_large = io.imread(filepath + filename)

if rescale_factor == 0:
    print('***AUTOMATIC RESCALING BASED ON FILE SIZE***')
    if composite_stack_large.nbytes >= 2.2e9:
        print("***FILE IS LARGER THAT 2 GB - DOWNSCALING BY 2 IN ALL DIMMENSIONS***")
        rescale_factor = 2
    else:
        print('***FILE IS SMALLER THAN 2 GB - NO RESCALING***')
        rescale_factor = 1

# Define pixel dimension to rescale factor
px_edge_rescaled = px_edge * rescale_factor

if rescale_factor > 1:
    composite_stack = np.asarray(StackResize(
        composite_stack_large, rescale_factor), dtype = 'uint8')
else:
     composite_stack = np.copy(composite_stack_large)   

print("***IDENTIFYING THE UNIQUE COLOR VALUES***")
unique_vals = np.unique(composite_stack)
print(unique_vals)

# Define color values
# This if..else statement needs to be cleaned up
if seg_values == 'default':
    mesophyll_value = 0
    if np.any(unique_vals == 85):
        stomata_value = 85
    elif np.any(unique_vals == 128):
        stomata_value = 128
    elif np.any(unique_vals == 152):
        stomata_value = 152
    bg_value = 177
    vein_value = 147
    ias_value = 255
    epidermis_ad_value = 30 if np.any(unique_vals == 30) else 60
    epidermis_ab_value = 60 if np.any(unique_vals == 60) else 69
    bs_value = 102
    vals_str = [mesophyll_value, stomata_value, bg_value, vein_value, ias_value,
                epidermis_ab_value, epidermis_ad_value, bs_value]
    # stomata_value = unique_vals[unique_vals not in vals_str] if stomata_value not in unique_vals else stomata_value
    # vals_str = [mesophyll_value, stomata_value, bg_value, vein_value, ias_value,
    #             epidermis_ab_value, epidermis_ad_value, bs_value]
    print("  Defined pattern values: ", str(vals_str))
    print("  Stomata value: ", str(stomata_value))
else:
    pix_values = [int(x) for x in seg_values.split(',')]
    # define pixel values
    mesophyll_value = pix_values[0]
    bg_value = pix_values[1]
    ias_value = pix_values[2]
    stomata_value = pix_values[3]
    epidermis_ad_value = pix_values[4]
    epidermis_ab_value = pix_values[5]
    vein_value = pix_values[6]
    bs_value = pix_values[7]
    vals_str = [mesophyll_value, stomata_value, bg_value, vein_value, ias_value,
                epidermis_ab_value, epidermis_ad_value, bs_value]
    print("  Defined pattern values: ", str(vals_str))

# Check if stomata have been labelled
if stomata_value not in unique_vals:
    if "stomata_stack_suffix" not in locals():
        print('************************************************')
        raise ValueError(sample_name + ': STOMATA HAVE NOT BEEN LABELLED!')

if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif'):
    print('***LOADING BOUNDING BOX CROPPED SEGMENTED STACK***')
    composite_stack = io.imread(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif')
    del composite_stack_large
    if 'stomata_stack_suffix' in locals():
        print('***LOADING BOUNDING BOX CROPPED STOMATA STACK***')
        stomata_stack = img_as_bool(io.imread(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATA_STACK_BBOX.tif'))
else:
    print("  Large stack shape: ", str(composite_stack_large.shape))
    print("  Small stack shape: ", str(composite_stack.shape))
    print("  Unique pattern values :", str(unique_vals))  # to get all the unique values

    # Remove the large stack to free up memory
    del composite_stack_large

    print('***CROPPING THE IMAGE AROUND THE BOUNDING BOX***')
    print('***         OF STOMATA AND EPIDERMIS         ***')
    rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(composite_stack, stomata_value)
    rminAD, rmaxAD, cminAD, cmaxAD, zminAD, zmaxAD = bbox2_3D(composite_stack, epidermis_ad_value)
    rminAB, rmaxAB, cminAB, cmaxAB, zminAB, zmaxAB = bbox2_3D(composite_stack, epidermis_ab_value)

    print("AD epidermis bbox:")
    print(rminAD, rmaxAD, cminAD, cmaxAD, zminAD, zmaxAD)
    print("AB epidermis bbox:")
    print(rminAB, rmaxAB, cminAB, cmaxAB, zminAB, zmaxAB)
    print("stomata bbox")
    print(rmin, rmax, cmin, cmax, zmin, zmax)

    print("  Small stack shape: ", str(composite_stack.shape))
    print("  Small stack nbytes: ", str(composite_stack.nbytes/1e9))
    print("   Bounding area:")
    if stomata_cropping:
        print("     slices:", zmin, zmax)
        print("          y:", cminAD, cmax)
        print("          x:", rmin, rmax)
        composite_stack = composite_stack[zmin:zmax, cminAD:cmax, rmin:rmax]
    else:
        print('    (only epidermis cropped - no stomata cropping)')
        print("          y:", cminAD, cmaxAB)
        composite_stack = composite_stack[:, cminAD:cmaxAB, :]
    print("  New shape: ", str(composite_stack.shape))
    print("  New nbytes: ", str(composite_stack.nbytes/1e9))

    print("***SAVING BOUNDING BOX CROPPED SEGMENTED STACK TO HARD DRIVE***")
    io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif', composite_stack)

    if 'stomata_stack_suffix' in locals():
        print('***LOADING INDEPENDENT STOMATA STACK***')
        stomata_stack = img_as_bool(io.imread(filepath + sample_name + stomata_stack_suffix))
        if stomata_cropping:
            stomata_stack = stomata_stack[zmin:zmax, cminAD:cmax, rmin:rmax]
        else:
            stomata_stack = stomata_stack[zmin:zmax, cminAD:cmaxAB, rmin:rmax]
        print("***SAVING BOUNDING BOX STOMATA STACK TO HARD DRIVE***")
        io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATA_STACK_BBOX.tif', stomata_stack)

# Create the binary stacks needed for the analysis
print('')
print('***CREATE BINARY STACKS***')
airspace_stack = Threshold(composite_stack, ias_value)
if 'stomata_stack' in locals():
    stomata_airspace_stack = airspace_stack + stomata_stack
    if not os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATA_AIRSPACE_STACK_BBOX.tif'):
        io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATA_AIRSPACE_STACK_BBOX.tif', stomata_airspace_stack)
else:
    if fix_stomata:
        print('***FILLING HOLES WITHIN LABELED STOMATA***')
        stomata_stack = Threshold(composite_stack, stomata_value)
        stomata_stack = binary_fill_holes(stomata_stack)
        stomata_stack = binary_dilation(stomata_stack)
        print("***SAVING FILLED HOLES STOMATA STACK TO HARD DRIVE***")
        io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATA_STACK_FILLED_HOLES_BBOX.tif', stomata_stack)
        stomata_airspace_stack = airspace_stack + stomata_stack
    else:
        stomata_airspace_stack = Threshold(composite_stack, [stomata_value, ias_value])

# Purify the airspace stack, i.e. get the largest connected component
print('***FINDING THE LARGEST AIRSPACE***')
# NOTE: Run 2nd line first and then define largest_Airsapce as True when within the one w stomata
largest_airspace = getLargestAirspace(airspace_stack)
largest_airspace_w_stomata = getLargestAirspace(stomata_airspace_stack)

# Removing not longer used array
del stomata_airspace_stack

print('***CROPPING THE LARGEST AIRSPACE STACK***')
rmin, rmax, cmin, cmax, zmin, zmax = bbox2_3D(largest_airspace_w_stomata, True)

print("  Largest airspace stack shape: ", str(largest_airspace.shape))
print("  Largest airspace stack nbytes: ", str(largest_airspace.nbytes/1e9))
print("  Bounding area:")
print("     slices:", zmin, zmax)
print("          y:", cmin, cmax)
print("          x:", rmin, rmax)
largest_airspace = largest_airspace[zmin:zmax, cmin:cmax, rmin:rmax]
largest_airspace_w_stomata = largest_airspace_w_stomata[zmin:zmax, cmin:cmax, rmin:rmax]
print("  New shape: ", str(largest_airspace.shape))
print("  New nbytes: ", str(largest_airspace.nbytes/1e9))

mask = ~largest_airspace.astype(bool)
if 'stomata_stack' in locals():
    stomata_stack = stomata_stack[zmin:zmax, cmin:cmax, rmin:rmax]
else:
    stomata_stack = np.asarray(Threshold(composite_stack, stomata_value), np.bool)
    stomata_stack = stomata_stack[zmin:zmax, cmin:cmax, rmin:rmax]
stom_mask = invert(stomata_stack)

# Cropping the composite stack to the largest arispace bounding box
composite_stack = composite_stack[zmin:zmax, cmin:cmax, rmin:rmax]

# Check if stomata stack does include values
# Will throw an error if at least one stomata is disconnected from the airspace
if np.sum(stomata_stack) == 0:
    print('ERROR: at least one stomata is disconnected from the airspace!')
    assert False

if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATAL_REGIONS_BBOX_CROPPPED.tif'):
    no_unique_stomata = True
else:
    if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'L_geo_BBOX_CROPPED.tif'):
        print('***LOADING PRECOMPUTED GEODESIC DISTANCE MAP***')
        L_geo = io.imread(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'L_geo_BBOX_CROPPED.tif')
        stomata_airspace_mask = ~largest_airspace_w_stomata.astype(bool)
        largest_airspace_masked_array = np.ma.masked_array(
            stom_mask, stomata_airspace_mask)
        max_L_geo = np.max(L_geo)
        L_geo = np.float32(L_geo)
    else:
        print('***COMPUTING GEODESIC DISTANCE MAP***')
        stomata_airspace_mask = ~largest_airspace_w_stomata.astype(bool)
        largest_airspace_masked_array = np.ma.masked_array(
            stom_mask, stomata_airspace_mask)
        t0 = time.time()
        L_geo = skfmm.distance(largest_airspace_masked_array)
        t1 = time.time() - t0
        print('  L_geo processing time: '+str(np.round(t1))+' s')
        L_geo = np.float32(L_geo)
        max_L_geo = np.max(L_geo)
        print('***SAVING GEODESIC DISTANCE MAP TO HARD DRIVE***')
        io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'L_geo_BBOX_CROPPED.tif', L_geo)

    # Create a rounded array of L_geo to find matches for each stoma
    L_geo_round = np.float32(np.round(L_geo, 0))

    print('***FINDING THE UNIQUE STOMATA REGIONS***')
    print('  this')
    # Make the stomata appear on a 2D surface, i.e. stomata positions
    stomata_pos_paradermal = np.sum(stomata_stack, axis=1)
    print('  may')
    unique_stoma = label(stomata_stack, connectivity=1)
    print('  take')
    props_of_unique_stoma = regionprops(unique_stoma)
    print('  a')
    stoma_centroid = np.zeros([len(props_of_unique_stoma), 3])
    print('  while')
    stomata_regions = np.zeros(stomata_stack.shape, dtype='uint8')

    print(' > There are ' + str(len(props_of_unique_stoma)) + ' stomata in this stack.')

    for regions in tqdm(np.arange(len(props_of_unique_stoma))):
        stoma_centroid[regions] = props_of_unique_stoma[regions].centroid
        stoma_mask = invert(unique_stoma == props_of_unique_stoma[regions].label)
        stoma_tmp_ma = np.ma.masked_array(stoma_mask, stomata_airspace_mask)
        if np.all(stoma_tmp_ma):
            # print("stoma",regions,"skipped because unconnected to airspace")
            continue
        t0 = time.time()
        L_geo_stom = skfmm.distance(stoma_tmp_ma, narrow=max_L_geo)
        L_geo_stom = np.float32(L_geo_stom)
        stomata_regions[np.round(L_geo_stom,0) == L_geo_round] = props_of_unique_stoma[regions].label
        t1 = time.time() - t0
        # print('  processing time: '+str(np.round(t1))+' s')
        del L_geo_stom
        del stoma_tmp_ma
        del stoma_mask
        gc.collect()

    regions_all = np.unique(stomata_regions)

    regions_at_border = np.unique(np.concatenate([np.unique(stomata_regions[0, :, :]),
                                                  np.unique(stomata_regions[-1, :, :]),
                                                  np.unique(stomata_regions[:, 0, :]),
                                                  np.unique(stomata_regions[:, -1, :]),
                                                  np.unique(stomata_regions[:, :, 0]),
                                                  np.unique(stomata_regions[:, :, -1])]))

    regions_full_in_center = regions_all[regions_at_border.take(
        np.searchsorted(regions_at_border, regions_all), mode='clip') != regions_all]

    full_stomata_regions_mask = np.empty(stomata_stack.shape, dtype='bool')
    for i in np.arange(len(regions_full_in_center)):
        full_stomata_regions_mask[stomata_regions
                                  == regions_full_in_center[i]] = True

    # DisplayRndSlices(full_stomata_regions_mask, 4)

    print('***SAVING THE UNIQUE STOMATAL REGIONS STACK***')
    io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATAL_REGIONS_BBOX_CROPPPED.tif',
              img_as_ubyte(stomata_regions*int(np.floor(255/max(regions_all)))))

    print('  Number of pixels in full stomatal regions: ' + \
        str(np.sum(full_stomata_regions_mask)))
    print('  Total number of airspace pixels: ' + str(np.sum(airspace_stack)))

    if np.sum(full_stomata_regions_mask) < 2000:
        print('***NO SINGLE STOMA REGIONS - too small high magnification stack?***')
        # If there are no single stomata regions, we still compute the values at the airspace edge.
        no_unique_stomata = True
        thefile = open(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'NO_SINGLE_STOMA_REGIONS.txt', 't')
        thefile.close()
    else:
        print('***EXTRACTING DATA FROM SINGLE STOMA REGIONS***')
        no_unique_stomata = False
        SA_single_region = np.empty(len(regions_full_in_center))
        Pore_volume_single_region = np.copy(SA_single_region)
        for regions in tqdm(np.arange(len(regions_full_in_center))):
            regions_bool = stomata_regions == regions_full_in_center[regions]
            ias_vert_faces = marching_cubes(regions_bool)
            ias_SA = mesh_surface_area(ias_vert_faces[0], ias_vert_faces[1])
            SA_single_region[regions] = ias_SA * (px_edge_rescaled**2)
            Pore_volume_single_region[regions] = np.sum(regions_bool) * (px_edge_rescaled**3)

        stoma_export_col_fix = int(np.floor(255/max(regions_all)))

        single_stoma_data = {"stoma_nb": regions_full_in_center,
                             "stoma_color_value": regions_full_in_center*stoma_export_col_fix,
                             "SA_single_region_um2": SA_single_region,
                             "Pore_volume_single_region_um3": Pore_volume_single_region
                             }

        if 'single_stoma_data' in locals():
            print('***EXPORTING SINGLE STOMA DATA TO TXT FILE***')
            full_stoma_out = DataFrame(single_stoma_data)
            full_stoma_out.to_csv(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SINGLE-STOMA-RESULTS.txt', sep='\t', encoding='utf-8')

# Detect edges of airspace
# Better to work on largest airspace as this is what is needed further down.
if os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'MESOPHYLL_EDGE_BBOX_CROPPPED.tif'):
    print('***LOADING THE OUTLINE OF THE AIRSPACE***')
    mesophyll_edge = img_as_bool(io.imread(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'MESOPHYLL_EDGE_BBOX_CROPPPED.tif'))
else:
    # This piece of code is really innefficent. I could be improved to be Faster
    # by not searching for all outline positions but only those near the epidermis.
    print('***CREATING THE OUTLINE OF THE AIRSPACE***')
    airspace_outline_smaller = Erosion3DimJ(largest_airspace)
    airspace_edge = largest_airspace ^ airspace_outline_smaller
    # io.imsave(filepath +  'STOMATA_and_TORTUOSITY/' + sample_name + '_airspace_edge.tif', img_as_ubyte(airspace_edge))
    del airspace_outline_smaller

    print('  Removing the airspace edge neighbour to the epidermis')
    print('  (this will take several minutes)')
    mesophyll_edge = np.zeros(airspace_edge.shape, dtype='bool')
    p = np.transpose(np.where(airspace_edge))
    shape = airspace_edge.shape
    bad_neighbours = joblib.Parallel(n_jobs=nb_cores)(joblib.delayed(get_bad_neighbours)
                                                      (p[i,], composite_stack, epidermis_ab_value,
                                                       epidermis_ad_value, shape)
                                                      for i in tqdm(np.arange(p.shape[0])))
    for i in tqdm(np.arange(p.shape[0])):
        mesophyll_edge[tuple(p[i])] = 0 if bad_neighbours[i] else 1
    io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'MESOPHYLL_EDGE_BBOX_CROPPPED.tif', img_as_ubyte(mesophyll_edge))

if 'full_stomata_regions_mask' not in locals():
    stomata_regions = io.imread(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'STOMATAL_REGIONS_BBOX_CROPPPED.tif')
    regions_all = np.unique(stomata_regions)
    regions_at_border = np.unique(np.concatenate([np.unique(stomata_regions[0, :, :]),
                                                  np.unique(stomata_regions[-1, :, :]),
                                                  np.unique(stomata_regions[:, 0, :]),
                                                  np.unique(stomata_regions[:, -1, :]),
                                                  np.unique(stomata_regions[:, :, 0]),
                                                  np.unique(stomata_regions[:, :, -1])]))
    regions_full_in_center = regions_all[regions_at_border.take(
        np.searchsorted(regions_at_border, regions_all), mode='clip') != regions_all]
    full_stomata_regions_mask = np.empty(stomata_stack.shape, dtype='bool')
    for i in np.arange(len(regions_full_in_center)):
        full_stomata_regions_mask[stomata_regions
                                  == regions_full_in_center[i]] = True

    if np.sum(full_stomata_regions_mask) < 2000:
        print('***NO SINGLE STOMA REGIONS - too small high magnification stack?***')
        # If there are no single stomata regions, we still compute the values at the airspace edge.
        no_unique_stomata = True
        thefile = open(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'NO_SINGLE_STOMA_REGIONS.txt', 'w')
        thefile.close()
    else:
        print('***EXTRACTING DATA FROM SINGLE STOMA REGIONS***')
        no_unique_stomata = False
        SA_single_region = np.empty(len(regions_full_in_center))
        Pore_volume_single_region = np.copy(SA_single_region)
        for regions in tqdm(np.arange(len(regions_full_in_center))):
            regions_bool = stomata_regions == regions_full_in_center[regions]
            ias_vert_faces = marching_cubes(regions_bool)
            ias_SA = mesh_surface_area(ias_vert_faces[0], ias_vert_faces[1])
            SA_single_region[regions] = ias_SA * (px_edge_rescaled**2)
            Pore_volume_single_region[regions] = np.sum(regions_bool) * (px_edge_rescaled**3)

        stoma_export_col_fix = int(np.floor(255/max(regions_all)))

        single_stoma_data = {"stoma_nb": regions_full_in_center,
                             "stoma_color_value": regions_full_in_center*stoma_export_col_fix,
                             "SA_single_region_um2": SA_single_region,
                             "Pore_volume_single_region_um3": Pore_volume_single_region
                             }

        if 'single_stoma_data' in locals():
            print('***EXPORTING SINGLE STOMA DATA TO TXT FILE***')
            full_stoma_out = DataFrame(single_stoma_data)
            full_stoma_out.to_csv(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'SINGLE-STOMA-RESULTS.txt', sep='\t', encoding='utf-8')

if no_unique_stomata:
    edge_and_full_stomata_mask = mesophyll_edge
else:
    edge_and_full_stomata_mask = mesophyll_edge & full_stomata_regions_mask

print('***SAVING THE STOMATAL REGIONS STACK***')
io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'MESOPHYLL_EDGE_AND_STOM_REGIONS_BBOX_CROPPPED.tif',
          img_as_ubyte(edge_and_full_stomata_mask))
    
if not os.path.isfile(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'L_Euc_BBOX_CROPPED.tif'):
    print('***COMPUTING EUCLIDIAN DISTANCE MAP***')
    t0 = time.time()
    L_euc = np.ma.masked_array(distance_transform_edt(stom_mask), mask, dtype="float32")
    t1 = time.time() - t0
    print('  L_euc processing time: '+str(np.round(t1))+' s')
    print('***SAVING EUCLIDIAN DISTANCE MAP TO HARD DRIVE***')
    io.imsave(filepath + 'STOMATA_and_TORTUOSITY/' + sample_name + 'L_Euc_BBOX_CROPPED.tif', L_euc)

t1 = time.time() - t_start
print('')
print('Total processing time: '+str(np.round(t1)/60)+' min')
print(sample_name + 'done!  (' + path_to_sample + ')')
