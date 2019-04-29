# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:07:35 2018

@author: Guillaume Theroux-Rancourt
"""


# Computing leaf tortuosity from segmented leaf microCT stacks
# Using the method in Earles et al. (2018)

# Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR
# (2018) Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact
# Mesophyll Conductance. (http://www.plantphysiol.org/content/178/1/148)
# Plant Physiol 178: 148-162


# __Created__ on 2018-03-21
# by Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
#
# __Last edited__ on 2019-03-15
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
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
import skfmm
import skimage.io as io
from skimage import img_as_ubyte
from skimage.util import invert
from skimage.measure import label, regionprops, marching_cubes_lewiner, mesh_surface_area
from skimage.transform import resize
import time
from tqdm import tqdm
import joblib
import multiprocessing

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
path_to_sample = str(sys.argv[1])
rescale_factor = int(sys.argv[2])
px_edge = float(sys.argv[3])
seg_values = sys.argv[4]
nb_cores = multiprocessing.cpu_count() if len(sys.argv) == 6 else int(sys.argv[5])
base_path = str(sys.argv[6])


# Function to resize in all 3 dimensions
# Loops over each slice: Faster and more memory efficient
# than working on the whole array at once.
def StackResize(stack, rf=rescale_factor):
    stack_rs = np.empty(np.array(stack.shape)/np.array([1, rf, rf]))
    for idx in np.arange(stack_rs.shape[0]):
        stack_rs[idx] = resize(stack[idx],
                               [stack.shape[1]/rf, stack.shape[2]/rf],
                               order=0, preserve_range=True)
    stack_rs2 = np.empty(np.array(stack_rs.shape)/np.array([rf, 1, 1]))
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


# Set directory of functions in order to import MLmicroCTfunctions
path_to_script = '/'.join(full_script_path.split('/')[:-1]) + '/'
# os.chdir(path_to_script)

sample_path_split = path_to_sample.split('/')
sample_name = sample_path_split[-2]
filename = sample_path_split[-1]
base_folder_name = base_path
filepath = base_folder_name + sample_name + '/'

px_edge_rescaled = px_edge * rescale_factor


# Check if file has already been processed
if os.path.isfile(filepath + sample_name + 'GEOMETRIC-TORTUOSITY-RESULTS.txt'):
    raise ValueError('This file has already been processed!')


# Read composite stack including slabelling of stomata
print('************************************************')
print('***STARTING GEOMETRIC TORTUOSITY ESTIMATES OF***')
print('            ' + sample_name)
print('')

# Check if file has already been processed
if os.path.isfile(filepath + sample_name + 'GEOMETRIC-TORTUOSITY-RESULTS.txt'):
    print('This file has already been processed!')
    assert False

print("***LOADING AND RESIZING STACK***")
composite_stack_large = io.imread(filepath + filename)
composite_stack = np.asarray(StackResize(
    composite_stack_large, rescale_factor), dtype='uint8')
unique_vals = np.unique(composite_stack)


print("  Large stack shape: ", str(composite_stack_large.shape))
print("  Small stack shape: ", str(composite_stack.shape))
print("  Unique pattern values :", str(unique_vals))  # to get all the unique values

del composite_stack_large

if seg_values == 'default':
    mesophyll_value = 0
    stomata_value = 85 if np.any(unique_vals == 85) else 128
    bg_value = 177
    vein_value = 147
    ias_value = 255
    epidermis_ad_value = 30 if np.any(unique_vals == 30) else 60
    epidermis_ab_value = 60 if np.any(unique_vals == 60) else 69
    bs_value = 102
    vals_str = [mesophyll_value, stomata_value, bg_value, vein_value, ias_value,
                epidermis_ab_value, epidermis_ad_value, bs_value]
    stomata_value = unique_vals[unique_vals not in vals_str] if stomata_value not in unique_vals else stomata_value
    vals_str = [mesophyll_value, stomata_value, bg_value, vein_value, ias_value,
                epidermis_ab_value, epidermis_ad_value, bs_value]
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


# Create the binary stacks needed for the analysis
print('')
print('***CREATE BINARY STACKS***')
airspace_stack = Threshold(composite_stack, ias_value)
stomata_airspace_stack = Threshold(composite_stack, [stomata_value, ias_value])

# Purify the airspace stack, i.e. get the largest connected component
print('***FINDING THE LARGEST AIRSPACE***')
largest_airspace = getLargestAirspace(airspace_stack)
largest_airspace_w_stomata = getLargestAirspace(stomata_airspace_stack)
mask = ~largest_airspace.astype(bool)

stomata_stack = np.asarray(Threshold(composite_stack, stomata_value), np.bool)
stom_mask = invert(stomata_stack)

# Check if stomata stack does include values
# Will throw an error if at least one stomata is disconnected from the airspace
if np.sum(stomata_stack) == 0:
    print('ERROR: at least one stomata is disconnected from the airspace!')
    assert False


# Detect edges of airspace
# Better to work on largest airspace as this is what is needed further down.
if os.path.isfile(filepath + sample_name + 'MESOPHYLL_EDGE.tif'):
    print('***LOADING THE OUTLINE OF THE AIRSPACE***')
    mesophyll_edge = io.imread(filepath + sample_name + 'MESOPHYLL_EDGE.tif')
else:
    # This piece of code is really innefficent. I could be improved to be Faster
    # by not searching for all outline positions but only those near the epidermis.

    print('***CREATING THE OUTLINE OF THE AIRSPACE***')
    airspace_outline_smaller = Erosion3DimJ(largest_airspace)
    airspace_edge = invert(Threshold(largest_airspace-airspace_outline_smaller, 0))
    # io.imsave(filepath + '_airspace_edge.tif', img_as_ubyte(airspace_edge))
    del airspace_outline_smaller

    print('  Removing the airspace edge neighbour to the epidermis')
    print('  (this will take several minutes)')
    edge_neighbours = np.zeros(airspace_edge.shape, dtype='uint8')
    mesophyll_edge = np.zeros(airspace_edge.shape, dtype='bool')
    p = np.transpose(np.where(airspace_edge))
    shape = airspace_edge.shape

    bad_neighbours = joblib.Parallel(n_jobs=nb_cores)(joblib.delayed(get_bad_neighbours)
                                                      (p[i, ], composite_stack, epidermis_ab_value,
                                                       epidermis_ad_value, shape)
                                                      for i in tqdm(np.arange(p.shape[0])))

    for i in tqdm(np.arange(p.shape[0])):
        mesophyll_edge[tuple(p[i])] = 0 if bad_neighbours[i] else 1

    io.imsave(filepath + sample_name + 'MESOPHYLL_EDGE.tif', img_as_ubyte(mesophyll_edge))


# ## Get the Euclidian distance from all stomata
print('***COMPUTING EUCLIDIAN DISTANCE MAP***')
t0 = time.time()
L_euc = np.ma.masked_array(distance_transform_edt(stom_mask), mask)
t1 = time.time() - t0
print('  L_euc processing time: '+str(np.round(t1))+' s')

L_euc_average_ax0 = np.mean(L_euc, axis=0)
L_euc_average_ax2 = np.mean(L_euc, axis=2)


# ## Get the geodesic distance map
#
# In the cell below, a purified/largest airspace stack needs to be used as an
# input as airspace unconnected to a stomata make the program run into an error
# (`ValueError: the array phi contains no zero contour (no zero level set)`).
#
# I initially ran into a error when trying to get a masked array to compute the
# distance map. The error I think was that stomata where often outside of the
# mask I was trying to impose, so an empty mask was produced. I solved that by
# creating an array with the stomata and the airspace together, and then
# creating the masked array of stomata position within the airspace+stomata stack.
#
# __Note:__ The airspace should be assigned a value of 1, and the stomata a
# value of 0. Cells and background should be white or have no value assigned.

stomata_airspace_mask = ~largest_airspace_w_stomata.astype(bool)

largest_airspace_masked_array = np.ma.masked_array(
    stom_mask, stomata_airspace_mask)

print('***COMPUTING GEODESIC DISTANCE MAP***')
t0 = time.time()
L_geo = skfmm.distance(largest_airspace_masked_array)
t1 = time.time() - t0
print('  L_geo processing time: '+str(np.round(t1))+' s')

L_geo_average = np.mean(L_geo, axis=0)

print('***COMPUTING TORTUOSITY FACTOR, TAU***')
Tortuosity_Factor = np.square(L_geo / L_euc)
Tortuosity_Factor[Tortuosity_Factor < 1] = 1
Tortuosity_factor_average_ax0 = np.mean(Tortuosity_Factor, axis=0)
Tortuosity_factor_average_ax2 = np.mean(Tortuosity_Factor, axis=2)

print('  Saving tortuosity as TIFF files')
io.imsave(filepath + sample_name + '-Python_tortuosity.tif',
          np.asarray(Tortuosity_Factor, dtype="float32"))
io.imsave(filepath + sample_name + '-Python_tortuosity_MEAN-ax0.tif',
          np.asarray(Tortuosity_factor_average_ax0, dtype="float32"))
io.imsave(filepath + sample_name + '-Python_tortuosity_MEAN-ax2.tif',
          np.asarray(Tortuosity_factor_average_ax2, dtype="float32"))

# ## Compute lateral diffusivity
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
else:
    mesophyll_stack = np.asarray(
        Threshold(composite_stack, [mesophyll_value, vein_value, ias_value, stomata_value]), np.bool)
    mesophyll_stack_shifted_up = np.roll(mesophyll_stack, -3, axis=1)
#    mesophyll_stack_shifted_down = np.roll(mesophyll_stack, 3, axis=1)
    epidermis_edge_bottom = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_up, 0)
#    epidermis_edge_top = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_down , 0)
#    amphistomatous_epidermis = Threshold(epidermis_edge_bottom + epidermis_edge_top, 1)

epidermis_edge_purified = getLargestAirspace(epidermis_edge_bottom)


# Compute L_epi
print('***COMPUTING L_EPI MAP***')
epidermis_mask = invert(epidermis_edge_purified)

t0 = time.time()
L_epi = np.ma.masked_array(distance_transform_edt(epidermis_mask), mask)
t1 = time.time() - t0

print('  L_epi processing time: '+str(np.round(t1, 1))+' s')

L_epi_average_ax0 = np.median(L_epi, axis=0)
L_epi_average_ax2 = np.median(L_epi, axis=2)

# Compute path lenthening.
# Uncomment the end to remove data close to the epidermis where lateral diffusivity values
print('***COMPUTING PATH LENGTH MAP***')
Path_lenghtening = (L_euc / L_epi)  # * (L_epi>10)


Path_lenghtening_average_ax0 = np.mean(Path_lenghtening, axis=0)
Path_lenghtening_average_ax2 = np.mean(Path_lenghtening, axis=2)


print('  Saving path length maps as TIFF files')
io.imsave(filepath + sample_name + '-Python_Path_lenghtening.tif',
          np.asarray(Path_lenghtening, dtype="float32"))
io.imsave(filepath + sample_name + '-Python_Path_lenghtening_MEAN_ax0.tif',
          np.asarray(Path_lenghtening_average_ax0, dtype="float32"))
io.imsave(filepath + sample_name + '-Python_Path_lenghtening_MEAN_ax2.tif',
          np.asarray(Path_lenghtening_average_ax2, dtype="float32"))


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

for regions in tqdm(np.arange(len(props_of_unique_stoma))):
    stoma_centroid[regions] = props_of_unique_stoma[regions].centroid
    L_euc_stom = np.ma.masked_array(distance_transform_edt(
        invert(unique_stoma == props_of_unique_stoma[regions].label)), mask)
    stomata_regions[L_euc_stom == L_euc] = props_of_unique_stoma[regions].label
    del L_euc_stom

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
io.imsave(filepath + sample_name + '-STOMATAL_REGIONS.tif',
          img_as_ubyte(stomata_regions*int(np.floor(255/max(regions_all)))))

print('  Number of pixels in full stomatal regions: ' + \
    str(np.sum(full_stomata_regions_mask)))
print('  Total number of airspace pixels: ' + str(np.sum(airspace_stack)))

if np.sum(full_stomata_regions_mask) < 10000:
    print('***NO SINGLE STOMA REGIONS - too small high magnification stack?***')
    # If there are no single stomata regions, we still compute the values at the airspace edge.
    edge_and_full_stomata_mask = mesophyll_edge
else:
    print('***EXTRACTING DATA FROM SINGLE STOMA REGIONS***')
    SA_single_region = np.empty(len(regions_full_in_center))
    Pore_volume_single_region = np.copy(SA_single_region)
    for regions in tqdm(np.arange(len(regions_full_in_center))):
        regions_bool = stomata_regions == regions_full_in_center[regions]
        ias_vert_faces = marching_cubes_lewiner(regions_bool)
        ias_SA = mesh_surface_area(ias_vert_faces[0], ias_vert_faces[1])
        SA_single_region[regions] = ias_SA * (px_edge_rescaled**2)
        Pore_volume_single_region[regions] = np.sum(regions_bool) * (px_edge_rescaled**3)

    stoma_export_col_fix = int(np.floor(255/max(regions_all)))

    single_stoma_data = {"stoma_nb": regions_full_in_center,
                         "stoma_color_value": regions_full_in_center*stoma_export_col_fix,
                         "SA_single_region_um2": SA_single_region,
                         "Pore_volume_single_region_um3": Pore_volume_single_region
                         }
    # Select only the values at the edge of the airspace and within the full stomata
    # Will have to find a way to include a larger zone of stomata
    edge_and_full_stomata_mask = mesophyll_edge & full_stomata_regions_mask

# np.where applies a condition to find True value, select those in an array
# (here values above or equal to 1, as tortuosity cannot be less than 1),
# and fills the False values with a specified value (here 0).
Tortuosity_at_airspace_edge = np.where(edge_and_full_stomata_mask == True,
                                       np.where(Tortuosity_Factor >= 1,
                                                Tortuosity_Factor, 0), 0)
Tortuosity_values_for_stats = Tortuosity_at_airspace_edge[Tortuosity_at_airspace_edge >= 1]

print("***TORTUOSITY VALUES AT THE AIRSPACE EDGE***")
print(np.nanmedian(Tortuosity_values_for_stats))
print(np.nanmean(Tortuosity_values_for_stats))
print(np.nanstd(Tortuosity_values_for_stats))
print(np.nanvar(Tortuosity_values_for_stats))
print(np.nanmin(Tortuosity_values_for_stats))
print(np.nanmax(Tortuosity_values_for_stats))
print('')

Path_lenghtening_at_airspace_edge = np.where(edge_and_full_stomata_mask == True,
                                             np.where(Path_lenghtening >= 1,
                                                      Path_lenghtening, 0), 0)
Path_lenghtening_values_for_stats = Path_lenghtening_at_airspace_edge[Path_lenghtening_at_airspace_edge >= 1]

print('***PATH LENGTH VALUES AT AIRSPACE EDGE***')
print('  median: ' + str(np.nanmedian(Path_lenghtening_values_for_stats)))
print(np.nanmean(Path_lenghtening_values_for_stats))
print(np.nanstd(Path_lenghtening_values_for_stats))
print(np.nanvar(Path_lenghtening_values_for_stats))
print(np.shape(Path_lenghtening_values_for_stats))
print(np.nanmin(Path_lenghtening_values_for_stats))
print(np.nanmax(Path_lenghtening_values_for_stats))
print('')

print('***COMPUTING SUMMARY VALUES***')
Path_lenght_at_airspace_edge_median = np.nanmedian(np.where(Path_lenghtening_at_airspace_edge != 0.,
                                                            Path_lenghtening_at_airspace_edge, np.nan), axis=0)

Tortuosity_at_airspace_edge_median = np.nanmedian(np.where(Tortuosity_at_airspace_edge != 0,
                                                           Tortuosity_at_airspace_edge, np.nan), axis=0)


Path_length_profile = np.nanmedian(Path_lenght_at_airspace_edge_median, axis=1)

Tortuosity_profile = np.nanmedian(Tortuosity_at_airspace_edge_median, axis=1)


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

print('***SAVING DATA***')
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
results_out.to_csv(filepath + sample_name + 'GEOMETRIC-TORTUOSITY-RESULTS.txt',
                   sep='\t', encoding='utf-8')

if 'single_stoma_data' in locals():
    full_stoma_out = DataFrame(single_stoma_data)
    full_stoma_out.to_csv(base_folder_name + sample_name + '/'
                          + sample_name + 'SINGLE-STOMA-RESULTS', sep='\t', encoding='utf-8')
else:
    thefile = open(filepath + sample_name + 'NO_SINGLE_STOMA_REGIONS.txt', 't')


# To save a txt file will all the data points
thefile = open(filepath + sample_name + '_Path_lenghtening_profile.txt', 'w')
for item in Path_length_profile:
    thefile.write("%s\n" % item)

thefile = open(filepath + sample_name + '_Tortuosity_profile.txt', 'w')
for item in Tortuosity_profile:
    thefile.write("%s\n" % item)

thefile = open(filepath + sample_name + '_SurfaceArea_profile.txt', 'w')
for item in surface_rel:
    thefile.write("%s\n" % item)

thefile = open(filepath + sample_name + '_Porosity_profile.txt', 'w')
for item in porosity_rel:
    thefile.write("%s\n" % item)


t1 = time.time() - t_start
print('')
print('Total processing time: '+str(np.round(t1)/60)+' min')
print(sample_name + 'done!  (' + path_to_sample + ')')
