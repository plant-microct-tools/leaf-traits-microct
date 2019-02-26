# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import joblib
import porespy as ps
import pytrax as pt
from skimage import io
import numpy as np
from skimage.transform import resize
from skimage.measure import label

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

def StackResize(stack, rescale_factor):
    stack_rs = np.empty(np.asarray(np.array(stack.shape)/[1,rescale_factor,rescale_factor],dtype='int64'), dtype='uint8')
    for idx in np.arange(stack_rs.shape[0]):
        stack_rs[idx] = resize(stack[idx], [stack.shape[1]/rescale_factor, stack.shape[2]/rescale_factor], order=0, preserve_range=True)
    stack_rs2 = np.empty(np.asarray(np.array(stack_rs.shape)/[rescale_factor,1,1],dtype='int64'), dtype='uint8')
    for idx in np.arange(stack_rs2.shape[1]):
        stack_rs2[:,idx,:] = resize(stack_rs[:,idx,:], [stack_rs.shape[0]/rescale_factor, stack_rs.shape[2]], order=0, preserve_range=True)
    return stack_rs2

def getLargestAirspace(input_img):
    # Label all the connected airspace
    # I specify here the background value, so we know which to remove later.
    # Connectivity=2 means 8-connected (faces+edges), =1 means 4-connected (faces only)
    labeled_img = label(input_img, background=0, connectivity=1)
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


def _get_tau(self, x, y, descriptor=None, color='k'):
    r'''
    Helper method to get tau
    '''
    a, res, _, _ = np.linalg.lstsq(x, y, rcond=-1)
    tau = 1/a[0]
    SStot = np.sum((y - y.mean())**2)
    rsq = 1 - (np.sum(res)/SStot)
    label = ('Tau: ' + str(np.around(tau, 3)) +
             ', R^2: ' + str(np.around(rsq, 3)))
    print(label)
    self.data[descriptor + '_tau'] = tau
    self.data[descriptor + '_rsq'] = rsq

img = io.imread('tort2d2.tif')
io.imshow(img)

rw = pt.RandomWalk(img)

rw.run(100000, nw=20000, stride=100, num_proc=4)

rw._add_linear_plot()
rw.calc_msd()

rw.plot_msd()

rw.axial_msd



im = ps.generators.blobs(shape=[300], porosity=0.5, blobiness=[1, 2, 1]).astype(bool)

io.imshow(im[100])

rw_blob = pt.RandomWalk(im)
rw.run(nt=1e4, nw=1e4, same_start=False, stride=100, num_proc=6)

rw.plot_msd()

# To get only the values, not the plot
x = np.arange(0, rw.nt, rw.stride)[:, np.newaxis]
a, res, _, _ = np.linalg.lstsq(x, rw.msd, rcond=-1)
tau = 1/a[0]
SStot = np.sum((rw.msd - rw.msd.mean())**2)
rsq = 1 - (np.sum(res)/SStot)
label = ('Tau: ' + str(np.around(tau, 3)) +
         ', R^2: ' + str(np.around(rsq, 3)))
print(label)

for ax in range(rw.dim):
    print('Axis ' + str(ax) + ' Square Displacement Data:')
    data = rw.axial_msd[:, ax]*rw.dim
    x = np.arange(0, rw.nt, rw.stride)[:, np.newaxis]
    a, res, _, _ = np.linalg.lstsq(x, data, rcond=-1)
    tau = 1/a[0]
    SStot = np.sum((data - data.mean())**2)
    rsq = 1 - (np.sum(res)/SStot)
    label = ('Tau: ' + str(np.around(tau, 3)) +
             ', R^2: ' + str(np.around(rsq, 3)))
    print(label)




# Test with real leaves

file_list = ['S_I_2_Strip3_','C_I_5_Strip1_']

# define pixel values
mesophyll_value = 0
stomata_value = 85
bg_value = 177
vein_value = 147
ias_value = 255
epidermis_ad_value = 30
epidermis_ab_value = 60

# Set the number of walkers
total_walkers = int(0.5e5)
walk_time = 100000

for iname in range(len(file_list)):
    # Set path to tiff stacks
    base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/_ML_DONE/'
    sample_name = file_list[iname]
    filepath = base_folder_name + sample_name + '/'
    rescale_factor = 2
    composite_stack_large = io.imread(filepath + sample_name + 'SEGMENTED-w-stomata.tif')
    composite_stack = StackResize(composite_stack_large, rescale_factor)
    print(composite_stack_large.shape)
    print(composite_stack.shape)
    print(np.unique(composite_stack)) # to get all the unique values
    del composite_stack_large

    airspace_stack = np.asarray(Threshold(composite_stack, ias_value), np.bool)
    stomata_airspace_stack = np.asarray(Threshold(composite_stack, [stomata_value,ias_value]), np.bool)

    # Purify the airspace stack, i.e. get the largest connected component
    largest_airspace = getLargestAirspace(airspace_stack)
#    largest_airspace_w_stomata = getLargestAirspace(stomata_airspace_stack)

    # Get the distance maps
    mask = ~largest_airspace.astype(bool)

#    stomata_stack = np.asarray(Threshold(composite_stack, stomata_value), np.bool)
#    stom_mask = invert(stomata_stack)
#
#    stomata_stack_shifted_down = np.roll(stomata_stack, 1, axis=1)
#    stomata_top_surface = Threshold(invert(stomata_stack) + stomata_stack_shifted_down , 0)
#
#    stomata_pos_paradermal = np.sum(stomata_top_surface, axis=1)
#    print(io.imshow(stomata_pos_paradermal))

    rw = pt.RandomWalk(composite_stack)
    rnd_walk_test = joblib.Parallel(n_jobs=6)(joblib.delayed(rw.run)(nt=5e4, nw=i, same_start=False, stride=10, num_proc=1) for i in tqdm(np.arange(1,11)*10000))
    joblib.dump(rnd_walk_test, filepath + 'Pytrax_test.joblib', compress='zlib')
    print('***RND WALK FROM RANDOM POINTS DONE***')
