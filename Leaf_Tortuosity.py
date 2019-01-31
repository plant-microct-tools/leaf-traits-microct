
# coding: utf-8

# In[ ]:


# Computing leaf tortuosity from segmented leaf microCT stacks
## Using the method in Earles et al. (to be submitted)

#__Created__ on 2018-03-21 by Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
#
#__Last edited__ on 2019-01-28 by Guillaume Théroux-Rancourt
#
#Image processing note:
#- The file used needs to have stomata either drawn on the segmented image (as is currently used) or on a separate image of the same dimensions as the segemented image (not implemented but easily done).
#- __How I have drawn the stomata__: How I did label the stomata was in ImageJ, using both the grid and segmented stacks. I replaced both so to get a paradermal view, and then synchronized both windows (Analyze > Tools > Synchronize windows). When I saw a stoma, I labelled it at or immediately below start of the IAS at that point. For label, I just use an ellipsis, but any form would do. I saved that ROI into the ROI manager, and proceeded to the other stomata. When I labelled all at their right location, I filled all ROI with a specific color on the segmented stack in paradermal view ([using this macro](https://github.com/gtrancourt/imagej_macros/blob/master/macros/fillInside-macro.txt)). I then replaced it to the former view (i.e. in te same direction as to get the paradermal view).
#
#Notes:
#- This code works and the results are comparable to what would be done using the MorpholibJ plugin in ImageJ. However, the results are not identical, probably due to the different implementations of the geodesic distance computation.
#
#To do:
#- Clean up the import packages.
#- Remove unused functions.
#- Create a nicer output of the results.
#
#Most recent updates:
#
#__2019-01-30
#-Stomate region of influence measure
#
#__2018-04-27__
#- Added stomata lbelling method in the preamble.
#- Splitted cells here and there to separate some bits.
#- Moved older updates to README.md
#
#I've got help and inspiration from:
#- https://stackoverflow.com/questions/28187867/geodesic-distance-transform-in-python
#- https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image


# In[1]:

import numpy as np
import os
from scipy import stats
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import skfmm
import skimage.io as io
from skimage import img_as_int, img_as_ubyte, img_as_float
from skimage.util import invert
from skimage.measure import label, regionprops
from skimage.transform import resize
import time
#import cv2
#import random


os.chdir('/home/gtrancourt/Dropbox/_github/microCT-leaf-traits/')
from Leaf_Tortuosity_Functions import DisplayRndSlices, Threshold, Erosion3DimJ, getLargestAirspace



# Function to resize in all 3 dimensions
# Loops over each slice: Faster and more memory efficient 
# than working on the whole array at once.
def StackResize(stack, rescale_factor):
    stack_rs = np.empty(np.array(stack.shape)/np.array([1,rescale_factor,rescale_factor]))
    for idx in np.arange(stack_rs.shape[0]):
        stack_rs[idx] = resize(stack[idx], [stack.shape[1]/rescale_factor, stack.shape[2]/rescale_factor], order=0, preserve_range=True)
    stack_rs2 = np.empty(np.array(stack_rs.shape)/np.array([rescale_factor,1,1]))
    for idx in np.arange(stack_rs2.shape[1]):
        stack_rs2[:,idx,:] = resize(stack_rs[:,idx,:], [stack_rs.shape[0]/rescale_factor, stack_rs.shape[2]], order=0, preserve_range=True)
    return stack_rs2



# ### Image Loading and Pre-processing

# In[2]:


# Set path to tiff stacks
base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/_ML_DONE/'
sample_name = 'S_I_2_Strip3_'
filepath = base_folder_name + sample_name + '/'

rescale_factor = 2

# In[3]:


# Read composite stack including slabelling of stomata
composite_stack_large = io.imread(filepath + sample_name + 'SEGMENTED-w-stomata.tif')
composite_stack = np.asarray(StackResize(composite_stack_large, rescale_factor), dtype='uint8')

print(composite_stack_large.shape)
print(composite_stack.shape)
print(np.unique(composite_stack)) # to get all the unique values

DisplayRndSlices(composite_stack, 4)

del composite_stack_large


# In[4]:


# define pixel values
mesophyll_value = 0
stomata_value = 85
bg_value = 177
vein_value = 147
ias_value = 255
epidermis_ad_value = 30
epidermis_ab_value = 60

# In[5]:


# Create the binary stacks needed for the analysis
#mesophyll_wo_veins_stack = np.asarray(Threshold(composite_stack, [mesophyll_value,ias_value,stomata_value]), np.bool)
#mesophyll_stack = np.asarray(Threshold(composite_stack, [mesophyll_value,vein_value,ias_value,stomata_value]), np.bool)
#cell_stack_w_veins = np.asarray(Threshold(composite_stack, [mesophyll_value,vein_value]), np.bool)
#airspace_stack = np.asarray(Threshold(composite_stack, ias_value), np.bool)
#stomata_stack = np.asarray(Threshold(composite_stack, stomata_value), np.bool)
#stomata_airspace_stack = np.asarray(Threshold(composite_stack, [stomata_value,ias_value]), np.bool)
#print(np.unique(stomata_stack))
#print(airspace_stack.dtype)
#print(stomata_stack.dtype)
#
#DisplayRndSlices(airspace_stack, 2)
#DisplayRndSlices(stomata_airspace_stack, 2)
#DisplayRndSlices(cell_stack_w_veins, 2)
#DisplayRndSlices(mesophyll_wo_veins_stack, 2)
#DisplayRndSlices(mesophyll_stack, 2)


# In[6]:

airspace_stack = np.asarray(Threshold(composite_stack, ias_value), np.bool)
stomata_airspace_stack = np.asarray(Threshold(composite_stack, [stomata_value,ias_value]), np.bool)

# Purify the airspace stack, i.e. get the largest connected component
largest_airspace = getLargestAirspace(airspace_stack)
largest_airspace_w_stomata = getLargestAirspace(stomata_airspace_stack)

DisplayRndSlices(largest_airspace, 2)
DisplayRndSlices(largest_airspace_w_stomata, 2)


# In[7]:


# Detect edges of airspace
# Better to work on largest airspace as this is what is needed further down.
airspace_outline_smaller = Erosion3DimJ(largest_airspace)
airspace_edge = invert(Threshold(largest_airspace - airspace_outline_smaller, 0))
DisplayRndSlices(airspace_edge, 2)
# io.imsave(filepath + '_airspace_edge.tif', img_as_ubyte(airspace_edge))
del airspace_outline_smaller

# In[8]:


# Get the distance maps
mask = ~largest_airspace.astype(bool)

stomata_stack = np.asarray(Threshold(composite_stack, stomata_value), np.bool)
stom_mask = invert(stomata_stack)

# Check if stomata stack does include values
# Will throw an error if at least one stomata is disconnected from the airspace
if np.sum(stomata_stack) == 0:
    print('ERROR: at least one stomata is disconnected from the airspace!')
    assert False

print(np.sum(stomata_stack))
DisplayRndSlices(mask, 2)
DisplayRndSlices(stom_mask, 2)


# ## Get the Euclidian distance from all stomata

# In[9]:


t0 = time.time()
L_euc = np.ma.masked_array(distance_transform_edt(stom_mask), mask)
t1 = time.time() - t0
print('L_euc processing time: '+str(np.round(t1))+' s')
DisplayRndSlices(L_euc, 2)

L_euc_average = np.mean(L_euc, axis=0)
io.imshow(L_euc_average)

# ## Get the geodesic distance map
#
# In the cell below, a purified/largest airspace stack needs to be used as an input as airspace unconnected to a stomata make the program run into an error (`ValueError: the array phi contains no zero contour (no zero level set)`).
#
# I initially ran into a error when trying to get a masked array to compute the distance map. The error I think was that stomata where often outside of the mask I was trying to impose, so an empty mask was produced. I solved that by creating an array with the stomata and the airspace together, and then creating the masked array of stomata position within the airspace+stomata stack.
#
# __Note:__ The airspace should be assigned a value of 1, and the stomata a value of 0. Cells and background should be white or have no value assigned.

# In[10]:


stomata_airspace_mask = ~largest_airspace_w_stomata.astype(bool)

largest_airspace_masked_array = np.ma.masked_array(stom_mask, stomata_airspace_mask)
DisplayRndSlices(largest_airspace_masked_array, 2)


# In[11]:


t0 = time.time()
L_geo = skfmm.distance(largest_airspace_masked_array)
t1 = time.time() - t0
print('L_geo processing time: '+str(np.round(t1))+' s')

DisplayRndSlices(L_geo, 2)



# ## Compute the tortuosity factor

# In[13]:


Tortuosity_Factor = np.square(L_geo / L_euc)
DisplayRndSlices(Tortuosity_Factor, 2)

Tortuosity_factor_average = np.mean(Tortuosity_Factor, axis=0)
io.imshow(Tortuosity_factor_average)

# You can save it to you folder by un-commenting the line below.
#io.imsave(filepath + 'Python_tortuosity.tif', np.asarray(Tortuosity_Factor, dtype="float32"))

#%%

# To analyse tortuosity in full stomatal regions (i.e. regions influenced by a
# single stomata and not touching the edges, meaning they are complete), we
# need first to find full regions. To do this, we need to do some spatial 
# analysis such as a Voronoi diagram. The best seems to be a KDTree 
# (nearest neighbour), like explained here: https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/spatial.html#voronoi-diagrams
# The KDTree function allows to query  filter to find the regions around the stomata. Then, within those regions,
# compute summary statistics for tortuosity and lateral diffusivity.

# Make the stomata appear on a 2D surface, i.e. stomata positions
stomata_pos_paradermal = np.sum(stomata_stack, axis=1)
io.imshow(stomata_pos_paradermal)
unique_stoma = label(stomata_stack, connectivity=1)
props_of_unique_stoma = regionprops(unique_stoma)
stoma_centroid = np.zeros([len(props_of_unique_stoma),3])
stomata_regions = np.zeros(stomata_stack.shape, dtype = 'uint8')
for regions in np.arange(len(props_of_unique_stoma)):
    stoma_centroid[regions] = props_of_unique_stoma[regions].centroid
    L_euc_stom = np.ma.masked_array(distance_transform_edt(invert(unique_stoma==props_of_unique_stoma[regions].label)), mask)
    stomata_regions[L_euc_stom == L_euc] = props_of_unique_stoma[regions].label
    del L_euc_stom

regions_all = np.unique(stomata_regions)

regions_at_border = np.unique(np.concatenate([np.unique(stomata_regions[0,:,:]),
                                              np.unique(stomata_regions[-1,:,:]),
                                              np.unique(stomata_regions[:,0,:]),
                                              np.unique(stomata_regions[:,-1,:]),
                                              np.unique(stomata_regions[:,:,0]),
                                              np.unique(stomata_regions[:,:,-1])]))

regions_full_in_center = regions_all[regions_at_border.take(np.searchsorted(regions_at_border, regions_all), mode='clip') != regions_all]
 
full_stomata_regions_mask = np.empty(stomata_stack.shape, dtype='bool')
for i in np.arange(len(regions_full_in_center)):
    full_stomata_regions_mask[stomata_regions == regions_full_in_center[i]] = True


# In[14]:


airspace_edge_bool = invert(~airspace_edge.astype(bool))

# Select only the values at the edge of the airspace
Tortuosity_at_mesophyll_surface = Tortuosity_Factor[full_stomata_regions_mask & airspace_edge_bool]
#Tortuosity_at_mesophyll_surface = Tortuosity_Factor[airspace_edge_bool]
Tortuosity_at_mesophyll_surface = Tortuosity_at_mesophyll_surface.filled([-1])

# Remove values below 1, if any, as tortuosity must be >= 1
Tortuosity_at_mesophyll_surface = Tortuosity_at_mesophyll_surface[Tortuosity_at_mesophyll_surface > 1]

print(np.median(Tortuosity_at_mesophyll_surface))
print(stats.describe(Tortuosity_at_mesophyll_surface))
print(np.mean(Tortuosity_at_mesophyll_surface))
print(np.std(Tortuosity_at_mesophyll_surface))
print(np.var(Tortuosity_at_mesophyll_surface))
print(np.shape(Tortuosity_at_mesophyll_surface))
print(np.min(Tortuosity_at_mesophyll_surface))
print(np.max(Tortuosity_at_mesophyll_surface))



# In[15]:


# To save
io.imsave(filepath + 'Python_tortuosity.tif', np.asarray(Tortuosity_Factor * airspace_edge_bool, dtype="float32"))


# In[16]:


# To save all the data point to a text file for later analysis
thefile = open(filepath + 'Tortuosity_at_mesophyll_surface.txt', 'w')
for item in Tortuosity_at_mesophyll_surface:
    thefile.write("%s\n" % item)


# ## Compute lateral diffusivity

# In[17]:

# To get the _abaxial_ epidermis layer as a single line
epidermis_ab_stack = np.asarray(Threshold(composite_stack, epidermis_ab_value), np.bool)
epidermis_ab_stack_shifted_down = np.roll(epidermis_ab_stack, 1, axis=1)
epidermis_edge_bottom = Threshold(invert(epidermis_ab_stack) + epidermis_ab_stack_shifted_down , 0)
DisplayRndSlices(epidermis_edge_bottom, 2)


# Get the epidermal layer map
#mesophyll_stack = np.asarray(Threshold(composite_stack, [mesophyll_value,vein_value,ias_value,stomata_value]), np.bool)
#
#mesophyll_stack_shifted_up = np.roll(mesophyll_stack, -1, axis=1)
#mesophyll_stack_shifted_down = np.roll(mesophyll_stack, 1, axis=1)
#epidermis_edge_bottom = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_up , 0)
#epidermis_edge_top = Threshold(invert(mesophyll_stack) + mesophyll_stack_shifted_down , 0)
#amphistomatous_epidermis = Threshold(epidermis_edge_bottom + epidermis_edge_top, 1)
#DisplayRndSlices(epidermis_edge_bottom, 1)
#DisplayRndSlices(epidermis_edge_top, 1)
#DisplayRndSlices(amphistomatous_epidermis, 1)


# In[18]:


# Compute L_epi
epidermis_mask = invert(epidermis_edge_bottom)
DisplayRndSlices(epidermis_mask, 2)

t0 = time.time()
L_epi = np.ma.masked_array(distance_transform_edt(epidermis_mask), mask)
t1 = time.time() - t0

print('L_epi processing time: '+str(np.round(t1, 1))+' s')
DisplayRndSlices(L_epi, 2)

L_epi_average = np.mean(L_epi, axis=0)
io.imshow(L_epi_average)

# In[19]:

# Compute path lenthening.
# Uncomment the end to remove data close to the epidermis where lateral diffusivity values 
Path_lenghtening = (L_euc / L_epi) #*(L_epi>10)
DisplayRndSlices(Path_lenghtening, 2)

Path_lenghtening_average = np.mean(Path_lenghtening, axis=0)
io.imshow(Path_lenghtening_average)

# In[20]:


Path_lenghtening_at_airspace_edge = Path_lenghtening[full_stomata_regions_mask & airspace_edge_bool]
Path_lenghtening_at_airspace_edge = Path_lenghtening_at_airspace_edge[Path_lenghtening_at_airspace_edge > 1]
print(np.median(Path_lenghtening_at_airspace_edge))
print(stats.describe(Path_lenghtening_at_airspace_edge))
print(np.nanmean(Path_lenghtening_at_airspace_edge))
print(np.nanstd(Path_lenghtening_at_airspace_edge))
print(np.nanvar(Path_lenghtening_at_airspace_edge))
print(np.shape(Path_lenghtening_at_airspace_edge))
print(np.nanmin(Path_lenghtening_at_airspace_edge))
print(np.nanmax(Path_lenghtening_at_airspace_edge))

Path_lengthening_profile = np.mean(Path_lenghtening*airspace_edge_bool, axis=(0,2))

# In[21]:


## To save a stack of lateral diffusivity at the airspace edge
Path_lenghtening_img = Path_lenghtening
Path_lenghtening_img[airspace_edge_bool == 0] = 0
io.imsave(filepath + 'Python_Path_lenghtening-3.tif', np.asarray(Path_lenghtening_img, dtype="float32"))


# In[22]:


# To save a txt file will all the data points
thefile = open(filepath + 'Path_lenghtening_at_airspace_edge.txt', 'w')
for item in Path_lenghtening_at_airspace_edge:
    thefile.write("%s\n" % item)


#%%





#
#                                   
#stoma_centroid_rounded = np.round(stoma_centroid)
#
#tree = KDTree(stoma_centroid_rounded)
#
#zz, xx, yy = np.meshgrid(np.arange(stomata_stack.shape[0]),
#                            np.arange(stomata_stack.shape[1]),
#                            np.arange(stomata_stack.shape[2]))
#zxy = np.c_[zz.ravel(), xx.ravel(), yy.ravel()]
#
#zxy = np.ndindex(stomata_stack.shape)
#
#stomata_regions = np.empty(stomata_stack.shape, dtype='uint8')
#for i in zxy:
#    tmp_ind = next(zxy)
#    stomata_regions[tmp_ind] = tree.query([tmp_ind])[1][0]
#
#tree.query([880,385,1182])[1]
#
#
##
### Make a Voronoi diagram
#stomata_pos_paradermal = np.sum(stomata_stack, axis=1)
#io.imshow(stomata_pos_paradermal)
#unique_stoma = label(stomata_pos_paradermal, connectivity=1)
#props_of_unique_stoma = regionprops(unique_stoma)
#stoma_centroid = np.zeros([len(props_of_unique_stoma),2])
#for regions in np.arange(len(props_of_unique_stoma)):
#    stoma_centroid[regions] = props_of_unique_stoma[regions].centroid
#
#vor = Voronoi(stoma_centroid)
#vor_fig = voronoi_plot_2d(vor)
##
##np.where(vor.point_region == 2)[0][0]
##
##regions, vertices = voronoi_finite_polygons_2d(vor)
##print "--"
##print regions
##print "--"
##print vertices
##