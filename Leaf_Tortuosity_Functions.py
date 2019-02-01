import random
import numpy as np
import skimage.io as io
from skimage.measure import label
from scipy import ndimage

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
            for i in range(len(Th_value)):
                tmp[input_img == Th_value[i]] = 1
    return tmp

# Randomly displays a specified number of slices.
def DisplayRndSlices(input_img, nb_of_slices=2):
    for i in random.sample(range(input_img.shape[0]), nb_of_slices):
        io.imshow(input_img[i,:,:])
        io.show()


# EdgeDetector3D detects the edge and is not equivalent to ImageJ's Binary > Outline if used on the airspace stack
# There's a Gaussian filter beging applied (sigma=). Using a low value doesn't necessarily close the edges.
# So, a better function should be implkemented
# Speed is ok but there is room for improvement
# see here for potential solution: https://stackoverflow.com/questions/29434533/edge-detection-for-image-stored-in-matrix
def EdgeDetector3D(input_img):
    tmp = np.zeros(input_img.shape)
    for i in range(input_img.shape[0]):
        tmp[i,:,:] = feature.canny(input_img[i,:,:], sigma=0.33)
    return tmp

# Better solution I found so far, but the edge is 2 pixel wide, i.e. going over both the airspace and non-airspace
# from https://stackoverflow.com/questions/29434533/edge-detection-for-image-stored-in-matrix
# Assumes you have a binary image
# TO DO:
# - Find a way to have only one pixel wide
# - Get rid of creating a color image
# - Get it into a 8-bit. It seems to be stuck to int64
def Outline2D(input_img):
    img_raw = input_img
    img_zero = np.zeros_like(img_raw, dtype=np.uint8)
    img = np.zeros_like(img_raw, dtype=np.uint8)
    img[:,:] = 128
    img[ img_raw < 0.25 ] = 0
    img[ img_raw > 0.75 ] = 255
    # define "next to" - this may be a square, diamond, etc
    selem = morphology.square(2)
    # create masks for the two kinds of edges
    black_white_edges = (filters.rank.minimum(img, selem) == 0) & (filters.rank.maximum(img, selem) == 255)
    # create a color image
    img_result = np.dstack( [img_zero,img_zero,img_zero] )
    # assign colors to edge masks
    img_result[ black_white_edges, : ] = np.asarray( [ 0, 255, 0 ] )
    img_result = color.rgb2gray(img_result) * 255
    img_result = img_result.astype(np.uint8)
    return img_result

def Outline3D(input_img):
    tmp = np.zeros(input_img.shape)
    for i in range(input_img.shape[0]):
        tmp[i,:,:] = Outline2D(input_img[i,:,:])
    return tmp

# Erosion3DimJ does a 1-px erosion on each slice of an image, like in ImageJ
# Used to produce the outline like in ImageJ
def Erosion3DimJ(input_img):
    tmp = np.zeros(input_img.shape)
    for i in range(input_img.shape[0]):
        tmp[i,:,:] = ndimage.morphology.binary_erosion(input_img[i,:,:])
    return tmp

def Outline3DimJ(input_img, largest_airspace_img):
    tmp = np.zeros(input_img.shape)
    for i in range(input_img.shape[0]):
        tmp[i,:,:] = ndimage.morphology.binary_erosion(input_img[i,:,:])
    return invert(Threshold(largest_airspace_img - tmp, 0))

# Function below is modified from
# https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
def getLargestAirspace(input_img):
    # Label all the connected airspace
    # I specify here the background value, so we know which to remove later.
    # Connectivity=2 means 8-connected (faces+edges), =1 means 4-connected (faces only)
    labeled_img = label(input_img, background=0, connectivity=2)
    labels_index = np.column_stack((np.unique(labeled_img) ,
                                   np.bincount(labeled_img.flat)))
    # Get the label of the largest airspace
    labels_index_sort = labels_index[:,1].argsort()
    if labels_index_sort[-1] == 0:
        largest_airspace_label = labels_index_sort[-2]
    else:
        largest_airspace_label = labels_index_sort[-1]
    # Create a new image
    largest_airspace = (labeled_img == largest_airspace_label)
    return largest_airspace
