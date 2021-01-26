# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:07:35 2018

@author: Guillaume Theroux-Rancourt
"""


# Computing leaf tortuosity and path lengthening
# Using some methods presented in in Earles et al. (2018),
# as well as new methods by Guillaume Théroux-Rancourt

# Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR
# (2018) Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact
# Mesophyll Conductance. (http://www.plantphysiol.org/content/178/1/148)
# Plant Physiol 178: 148-162


# __Created__ on 2020-11-10
# by Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
#
# __Last edited__ on 2020-11-10
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
#
# TO DO
# - Integrate sampling in distance_transform_edt and in L_geo computation
# - Clean up the if..else statement for seg.values
# - Add nice input using dictionary as in the main repo for segmentation and trait analysis


import sys
import os
import numpy as np
from pandas import DataFrame
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
import skfmm
import skimage.io as io
from skimage import img_as_ubyte, img_as_bool
from skimage.util import invert
from skimage.measure import label
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



# Function to resize in all 3 dimensions
# Loops over each slice: Faster and more memory efficient
# than working on the whole array at once.
def StackResize(stack, rf=rescale_factor):
    resized_shape = np.array(stack.shape)/np.array([1, rf, rf])
    stack_rs = np.empty(shape = resized_shape.astype(np.int64))
    for idx in np.arange(stack_rs.shape[0]):
        stack_rs[idx] = resize(stack[idx],
                               [stack.shape[1]/rf, stack.shape[2]/rf],
                               order=0, preserve_range=True)
    resized_shape_2 = np.array(stack_rs.shape)/np.array([rf, 1, 1])
    stack_rs2 = np.empty(shape = resized_shape_2.astype(np.int64))
    for idx in np.arange(stack_rs2.shape[1]):
        stack_rs2[:, idx, :] = resize(stack_rs[:, idx, :],
                                      [stack_rs.shape[0]/rf, stack_rs.shape[2]],
                                      order=0, preserve_range=True)
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
    tmp = np.zeros(input_img.shape)
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

# TESTING
# path_to_sample = 'S12_Leaf2_1_/S12_Leaf2_1_SEGMENTED_w_STOMATA_BBOX.tif'
# base_path = '/run/media/guillaume/Elements/Vitis_Shade_Drought/2019/_TORT_TO_DO/'
# rescale_factor = 1

# Set directory of functions in order to import MLmicroCTfunctions
path_to_script = '/'.join(full_script_path.split('/')[:-1]) + '/'
base_folder_name = base_path
sample_path_split = path_to_sample.split('/')

if len(sample_path_split) == 1:
    sample_name = path_to_sample
    filename = sample_name + 'SEGMENTED.tif'
    filepath = base_folder_name + sample_name + '/'
    save_path = filepath + 'STOMATA_and_TORTUOSITY/'
elif len(sample_path_split) == 2:
    if sample_path_split[0] == '.':
        sample_name = sample_path_split[-1]
        if os.path.isfile(base_folder_name + sample_name + '/' + sample_name + 'SEGMENTED_w_STOMATA.tif'):
            filename = sample_name + 'SEGMENTED_w_STOMATA.tif'
        else:
            filename = sample_name + 'SEGMENTED.tif'
        filepath = base_folder_name + sample_name + '/'
        save_path = filepath + 'STOMATA_and_TORTUOSITY/'
    else:
        sample_name = sample_path_split[-2]
        filename = sample_path_split[-1]
        filepath = base_folder_name + sample_name + '/'
        save_path = filepath + 'STOMATA_and_TORTUOSITY/'
elif len(sample_path_split) == 3:
    if sample_path_split[0] == '.':
        sample_name = sample_path_split[-2]
        filename = sample_path_split[-1]
        filepath = base_folder_name + sample_name + '/'
        save_path = filepath + 'STOMATA_and_TORTUOSITY/'
    else:
        sample_name = sample_path_split[-3]
        base_folder_name = base_path + '/' + sample_name + '/' + sample_path_split[-2] + '/'
        filepath = base_folder_name
        filename = sample_path_split[-1]
        save_path = filepath + 'STOMATA_and_TORTUOSITY/'

print('Base folder path: ', base_folder_name)
print('Filepath: ', filepath)
# TESTING
# print('Save path: ', save_path)
# print(os.path.isfile(save_path + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif'))

px_edge_rescaled = px_edge * rescale_factor

# Check if file has already been processed
if os.path.isfile(save_path + sample_name + 'GEOMETRIC-TORTUOSITY-RESULTS.txt'):
    raise ValueError('This file has already been processed!')

# Check if the stomatal regions have been already identified
if not os.path.isfile(save_path + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif'):
    raise ValueError('The stomatal regions have not been identified!\nPlease run Leaf_Stomatal_Regions_py3.py first.')
else:
    # Read composite stack including slabelling of stomata
    print('************************************************')
    print('***STARTING TORTUOSITY AND PATH LENGTHENING COMPUTATION FOR***')
    print('   ' + sample_name)
    print('')
    # HERE WE ASSUME THAT THIS FILE HAS BEEN RESCALED IF NEEDED.
    # CURRENTLY WORKS ON NOT RESCALED STACKS (i.e. rescale=1)
    print('***LOADING BOUNDING BOX CROPPED SEGMENTED STACK***')
    composite_stack = io.imread(save_path + sample_name + 'SEGMENTED_w_STOMATA_BBOX.tif')
    # print('***LOADING PRECOMPUTED EUCLIDIAN DISTANCE MAP***')
    # L_euc = io.imread(filepath + sample_name + 'L_Euc_BBOX_CROPPED.tif')
    print('***LOADING PRECOMPUTED FULL STOMATAL REGIONS MESOPHYLL EDGE***')
    edge_and_full_stomata_mask = img_as_bool(io.imread(save_path + sample_name + 'MESOPHYLL_EDGE_AND_STOM_REGIONS_BBOX_CROPPPED.tif'))

print("***IDENTIFYING THE UNIQUE COLOR VALUES***")
unique_vals = np.unique(composite_stack)
print(unique_vals)

# Define color values
# TO DO This if..else statement needs to be cleaned up
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

if 'stomata_stack_suffix' in locals():
    print('***LOADING BOUNDING BOX CROPPED STOMATA STACK***')
    stomata_stack = img_as_bool(io.imread(save_path + sample_name + 'STOMATA_STACK_BBOX.tif'))

print('***CREATE BINARY STACKS***')
airspace_stack = Threshold(composite_stack, ias_value)
if 'stomata_stack' in locals():
    stomata_airspace_stack = airspace_stack + stomata_stack
else:
    stomata_airspace_stack = Threshold(composite_stack, [stomata_value, ias_value])

# Purify the airspace stack, i.e. get the largest connected component
print('***FINDING THE LARGEST AIRSPACE***')
largest_airspace = getLargestAirspace(airspace_stack)
largest_airspace_w_stomata = getLargestAirspace(stomata_airspace_stack)

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
airspace_stack = airspace_stack[zmin:zmax, cmin:cmax, rmin:rmax]
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

# Compute
if not os.path.isfile(save_path + sample_name + 'L_epi_BBOX_CROPPED.tif'):
    print('***MAP THE ABAXIAL EPIDERMIS***')
    # To get the _abaxial_ epidermis layer as a single line
    if epidermis_ab_value != epidermis_ad_value:
        epidermis_ab_stack = np.asarray(
            Threshold(composite_stack, epidermis_ab_value),
            np.bool)
        epidermis_ab_stack_shifted_down = np.roll(epidermis_ab_stack, 3, axis=1)
        epidermis_edge_bottom = Threshold(
            invert(epidermis_ab_stack) + epidermis_ab_stack_shifted_down, 0)
        epidermis_edge_bottom = epidermis_ab_stack
        del epidermis_ab_stack
        del epidermis_ab_stack_shifted_down

    else:
        mesophyll_stack = np.asarray(
            Threshold(composite_stack, [mesophyll_value, vein_value, ias_value, stomata_value]), np.bool)
        mesophyll_stack_shifted_up = np.roll(mesophyll_stack, -3, axis=1)
        #    mesophyll_stack_shifted_down = np.roll(mesophyll_stack, 3, axis=1)
        epidermis_edge_bottom = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_up, 0)
    #    epidermis_edge_top = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_down , 0)
    #    amphistomatous_epidermis = Threshold(epidermis_edge_bottom + epidermis_edge_top, 1)

    epidermis_edge_purified = getLargestAirspace(epidermis_edge_bottom)

    del epidermis_edge_bottom
    gc.collect()

    print('***COMPUTING L_EPI MAP***')
    epidermis_mask = invert(epidermis_edge_purified)
    del epidermis_edge_purified
    gc.collect()
    t0 = time.time()
    L_epi = np.ma.masked_array(distance_transform_edt(epidermis_mask), mask, dtype="float32")
    t1 = time.time() - t0
    print('  L_epi processing time: ' + str(np.round(t1, 1)) + ' s')
    print('***SAVING EPIDERMIS DISTANCE MAP TO HARD DRIVE***')
    io.imsave(save_path + sample_name + 'L_epi_BBOX_CROPPED.tif', L_epi)

    del L_epi
    del epidermis_mask
    gc.collect()

if os.path.isfile(save_path + sample_name + 'Python_tortuosity_BBOX_CROPPED.tif'):
    print('***LOADING PRECOMPUTED TORTUOSITY FACTOR***')
    Tortuosity_Factor = io.imread(save_path + sample_name + 'Python_tortuosity_BBOX_CROPPED.tif')
else:
    if not os.path.isfile(save_path + sample_name + 'L_geo_BBOX_CROPPED.tif'):
        print('***COMPUTING GEODESIC DISTANCE MAP***')
        stomata_airspace_mask = ~largest_airspace_w_stomata.astype(bool)
        largest_airspace_masked_array = np.ma.masked_array(
            stom_mask, stomata_airspace_mask)
        t0 = time.time()
        L_geo = skfmm.distance(largest_airspace_masked_array)
        t1 = time.time() - t0
        print('  L_geo processing time: '+str(np.round(t1))+' s')
        L_geo = np.float32(L_geo)
        print('***SAVING GEODESIC DISTANCE MAP TO HARD DRIVE***')
        io.imsave(save_path + sample_name + 'L_geo_BBOX_CROPPED.tif', L_geo)
    if not os.path.isfile(save_path + sample_name + 'L_Euc_BBOX_CROPPED.tif'):
        print('***COMPUTING EUCLIDIAN DISTANCE MAP***')
        t0 = time.time()
        L_euc = np.ma.masked_array(distance_transform_edt(stom_mask), mask, dtype="float32")
        t1 = time.time() - t0
        print('  L_euc processing time: ' + str(np.round(t1)) + ' s')
        print('***SAVING EUCLIDIAN DISTANCE MAP TO HARD DRIVE***')
        io.imsave(save_path + sample_name + 'L_Euc_BBOX_CROPPED.tif', L_euc)
    if os.path.isfile(save_path + sample_name + 'L_geo_BBOX_CROPPED.tif'):
        print('***LOADING PRECOMPUTED GEODESIC DISTANCE MAP***')
        L_geo = io.imread(save_path + sample_name + 'L_geo_BBOX_CROPPED.tif')
    if os.path.isfile(save_path + sample_name + 'L_Euc_BBOX_CROPPED.tif'):
        print('***LOADING PRECOMPUTED EUCLIDIAN DISTANCE MAP***')
        L_euc = io.imread(save_path + sample_name + 'L_Euc_BBOX_CROPPED.tif')
    print('***COMPUTING TORTUOSITY FACTOR, TAU***')
    Tortuosity_Factor = np.square(L_geo / L_euc)
    Tortuosity_Factor[Tortuosity_Factor < 1] = 1
    Tortuosity_factor_average_ax0 = np.mean(Tortuosity_Factor, axis=0)
    Tortuosity_factor_average_ax2 = np.mean(Tortuosity_Factor, axis=2)

    print('***SAVING TORTUOSITY MAP TO HARD DRIVE***')
    io.imsave(save_path + sample_name + 'Python_tortuosity_BBOX_CROPPED.tif',
              np.asarray(Tortuosity_Factor, dtype="float32"))
    io.imsave(save_path + sample_name + 'Python_tortuosity_MEAN-ax0.tif',
              np.asarray(Tortuosity_factor_average_ax0, dtype="float32"))
    io.imsave(save_path + sample_name + 'Python_tortuosity_MEAN-ax2.tif',
              np.asarray(Tortuosity_factor_average_ax2, dtype="float32"))

    # Remove L_geo to free up memory
    del L_geo
    del L_euc
    del Tortuosity_factor_average_ax0
    del Tortuosity_factor_average_ax2
    gc.collect()

# COMPUTING SUMMARY VALUES FOR TORTUOSITY
# np.where applies a condition to find True value, select those in an array
# (here values above or equal to 1, as tortuosity cannot be less than 1),
# and fills the False values with a specified value (here 0).
Tortuosity_at_airspace_edge = np.where(edge_and_full_stomata_mask == True,
                                       np.where(Tortuosity_Factor >= 1,
                                                Tortuosity_Factor, 0), 0)
Tortuosity_values_for_stats = Tortuosity_at_airspace_edge[Tortuosity_at_airspace_edge >= 1]

# To save a txt file will all the data points
thefile = open(save_path + sample_name + 'Tortuosity_values_for_stats.txt', 'w')
for item in Tortuosity_values_for_stats:
    thefile.write("%s\n" % item)
thefile.close()

print("***TORTUOSITY VALUES AT THE AIRSPACE EDGE***")
print('    median:',np.nanmedian(Tortuosity_values_for_stats))
print('      mean:',np.nanmean(Tortuosity_values_for_stats))
print('        sd:',np.nanstd(Tortuosity_values_for_stats))
print('       var:',np.nanvar(Tortuosity_values_for_stats))
print('       min:',np.nanmin(Tortuosity_values_for_stats))
print('       max:',np.nanmax(Tortuosity_values_for_stats))
print('')

Tortuosity_at_airspace_edge_median = np.nanmedian(np.where(Tortuosity_at_airspace_edge != 0,
                                                           Tortuosity_at_airspace_edge, np.nan), axis=0)
Tortuosity_profile = np.nanmedian(Tortuosity_at_airspace_edge_median, axis=1)

# To save as tif file will all the data points
io.imsave(save_path + sample_name + 'Tortuosity_at_airspace_edge_median.tif', Tortuosity_at_airspace_edge_median)

del Tortuosity_Factor
gc.collect()

# COMPUTE DISTANCE FROM EPIDERMIS MAP
if os.path.isfile(save_path + sample_name + 'Python_Path_lenghtening_BBOX_CROPPED.tif'):
    print('***LOADING PRECOMPUTED PATH LENGTHENING MAP***')
    Path_lenghtening = io.imread(save_path + sample_name + 'Python_Path_lenghtening_BBOX_CROPPED.tif')
else:
    print('***LOADING PRECOMPUTED EPIDERMIS DISTANCE MAP***')
    L_epi = io.imread(save_path + sample_name + 'L_epi_BBOX_CROPPED.tif')
    print('***LOADING PRECOMPUTED EUCLIDIAN DISTANCE MAP***')
    L_euc = io.imread(save_path + sample_name + 'L_Euc_BBOX_CROPPED.tif')
    print('***COMPUTING PATH LENGTHENING MAP***')
    Path_lenghtening = (L_euc / L_epi)  # * (L_epi>10)
    Path_lenghtening_average_ax0 = np.mean(Path_lenghtening, axis=0)
    Path_lenghtening_average_ax2 = np.mean(Path_lenghtening, axis=2)

    # Remove L_euc and L_epi
    del L_epi
    del L_euc
    gc.collect()

    print('  Saving path length maps as TIFF files')
    io.imsave(save_path + sample_name + 'Python_Path_lenghtening_BBOX_CROPPED.tif',
              np.asarray(Path_lenghtening, dtype="float32"))
    io.imsave(save_path + sample_name + 'Python_Path_lenghtening_MEAN_ax0.tif',
              np.asarray(Path_lenghtening_average_ax0, dtype="float32"))
    io.imsave(save_path + sample_name + 'Python_Path_lenghtening_MEAN_ax2.tif',
              np.asarray(Path_lenghtening_average_ax2, dtype="float32"))

Path_lenghtening_at_airspace_edge = np.where(edge_and_full_stomata_mask == True,
                                             np.where(Path_lenghtening >= 1,
                                                      Path_lenghtening, 0), 0)
Path_lenghtening_values_for_stats = Path_lenghtening_at_airspace_edge[Path_lenghtening_at_airspace_edge >= 1]

# To save a txt file will all the data points
thefile = open(save_path + sample_name + 'Path_lenghtening_values_for_stats.txt', 'w')
for item in Path_lenghtening_values_for_stats:
    thefile.write("%s\n" % item)
thefile.close()    

print('***PATH LENGTH VALUES AT AIRSPACE EDGE***')
print('  median: ', np.nanmedian(Path_lenghtening_values_for_stats))
print('    mean: ', np.nanmean(Path_lenghtening_values_for_stats))
print('      sd: ', np.nanstd(Path_lenghtening_values_for_stats))
print('     var: ', np.nanvar(Path_lenghtening_values_for_stats))
print('     min: ', np.nanmin(Path_lenghtening_values_for_stats))
print('     max: ', np.nanmax(Path_lenghtening_values_for_stats))
print('')

Path_lenght_at_airspace_edge_median = np.nanmedian(np.where(Path_lenghtening_at_airspace_edge != 0.,
                                                            Path_lenghtening_at_airspace_edge, np.nan), axis=0)
Path_length_profile = np.nanmedian(Path_lenght_at_airspace_edge_median, axis=1)

# To save as tif file will all the data points
io.imsave(save_path + sample_name + 'Path_length_at_airspace_edge_median.tif', Path_lenght_at_airspace_edge_median)
#io.imsave(filepath + sample_name + 'Path_length_profile.tif', Path_length_profile)

del Path_lenghtening
gc.collect()

# COMPUTE SUMMARY VALUES FOR EXPOSED SURFACE AND POROSITY
print('***COMPUTING SUMMARY VALUES FOR EXPOSED SURFACE AND POROSITY***')
mesophyll_edge = io.imread(save_path + sample_name + 'MESOPHYLL_EDGE_BBOX_CROPPPED.tif')

surface_cumsum = np.cumsum(np.sum(mesophyll_edge, axis=(0, 2)))
surface_rel = surface_cumsum/np.float(surface_cumsum.max())
surface_rel_ = surface_rel[surface_rel > 0]
pos_at_50_surface = (surface_rel_ >= 0.5).argmax()
surface_sum = np.sum(mesophyll_edge, axis=(0, 2))
surface_rel = surface_sum/np.float(surface_sum.max())

porosity_cumsum = np.cumsum(np.sum(airspace_stack, axis=(0, 2)))
porosity_rel = porosity_cumsum/np.float(porosity_cumsum.max())
porosity_rel_ = porosity_rel[surface_rel > 0]
pos_at_50_porosity = (porosity_rel_ >= 0.5).argmax()
porosity_sum = np.sum(airspace_stack, axis=(0, 2))
porosity_rel = porosity_sum/np.float(porosity_sum.max())

print('***SAVING RESULT FILES***')
# Write the data into a data frame
data_out = {'Tortuosity_MEAN': np.nanmean(Tortuosity_values_for_stats),
            'Tortuosity_MEDIAN': np.nanmedian(Tortuosity_values_for_stats),
            'Tortuosity_SD': np.std(Tortuosity_values_for_stats),
            'Tortuosity_VAR': np.var(Tortuosity_values_for_stats),
            'Tortuosity_SKEW': stats.skew(Tortuosity_values_for_stats),
            'Tortuosity_50percent_surface': Tortuosity_profile[pos_at_50_surface],

            'Path_lenghtening_MEAN': np.nanmean(Path_lenghtening_values_for_stats),
            'Path_lenghtening_MEDIAN': np.nanmedian(Path_lenghtening_values_for_stats),
            'Path_lenghtening_SD': np.std(Path_lenghtening_values_for_stats),
            'Path_lenghtening_VAR': np.var(Path_lenghtening_values_for_stats),
            'Path_lenghtening_SKEW': stats.skew(Path_lenghtening_values_for_stats),
            'Path_lenghtening_50percent_surface': Path_length_profile[pos_at_50_surface]}

results_out = DataFrame(data_out, index={sample_name})
# Save the data to a CSV
results_out.to_csv(save_path + sample_name + 'GEOMETRIC-TORTUOSITY-RESULTS.txt',
                   sep='\t', encoding='utf-8')

# To save a txt file will all the data points
thefile = open(save_path + sample_name + 'Path_lenghtening_profile.txt', 'w')
for item in Path_length_profile:
    thefile.write("%s\n" % item)
thefile.close()    

thefile = open(save_path + sample_name + 'Tortuosity_profile.txt', 'w')
for item in Tortuosity_profile:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open(save_path + sample_name + 'SurfaceArea_profile.txt', 'w')
for item in surface_rel:
    thefile.write("%s\n" % item)
thefile.close()

thefile = open(save_path + sample_name + 'Porosity_profile.txt', 'w')
for item in porosity_rel:
    thefile.write("%s\n" % item)
thefile.close()

t1 = time.time() - t_start
print('')
print('Total processing time: '+str(np.round(t1)/60)+' min')
print(sample_name + 'done!  (' + path_to_sample + ')')