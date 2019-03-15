# Tools to process and analyze plant leaf microCT scans

X-ray micro-computed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input. By segmentation we mean the identification of specific tissues within the leaves as single values within an image file.

We also provide further tools to process the segmented images, to extract leaf anatomical traits commonly measured, such as in [Théroux-Rancourt et al. (2017)](#references), or to compute airspace tortuosity and related airspace diffusion traits from [Earles et al. (2018)](#references).

<!-- ![alt text][logo]

[logo]: https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/imgs_readme/leaf1.png "translucent epidermis with veins" -->

This repository combines the most up-to-date code for the segmentation and leaf traits analysis. The leaf segmentation project has been initiated by [Mason Earles](https://github.com/masonearles/3DLeafCT) and expanded by [Matt Jenkins](https://github.com/mattjenkins3/3DLeafCT). The current version has been edited by [Guillaume Théroux-Rancourt](https://github.com/gtrancourt) with [these changes from the previous version](#changes-made-to-the-previous-version). All leaf traits analysis were contributed by Guillaume Théroux-Rancourt.


### Go directly to the procedure for specific tools
- [(Semi-)Automated leaf segmentation](#leaf-segmentation-leaf_segmentationpy)
- [Leaf traits analysis (and segmentation post-processing)](#post-processing-and-leaf-traits-analysis)
- [Leaf tortuosity and airspace diffusive traits analysis](#leaf-tortuosity-and-airspace-diffusive-traits-analysis)


## Requirements
- __python 2__: If you are new to python, a nice and convenient way to install python is through [anaconda](https://www.anaconda.com/download/). The code here uses python 2.7, so be sure to install this version. _Note that the code will eventually be ported to python 3 as python 2 will reach the end of its life in January 2020._ The anaconda navigator makes it easy to install packages, which can also be installed through command line in your terminal by typing `conda install name_of_package`. Further, the anaconda navigator allows you to open specific applications to write and run python code. Spyder is a good choice for scientific computation.
- __RAM:__ Processing a 5 Gb 8-bit multi-sliced tiff file can peak up to 60 GB of RAM and use up to 30-40 Gb of swap memory (memory written on disk) on Linux (and takes about 3-5 hours to complete). Processing a 250 Mb 8-bit file is of course a lot faster (30-90 minutes) but still takes about 10 Gb of RAM. The program is memory savvy and this needs to be addressed in future versions in order to tweak the program.


## Procedure

### Download the code to your computer

There are two ways to copy the code to your computer.

- [Clone the repository](https://help.github.com/articles/cloning-a-repository/): Using `git` will allow you to conveniently download the latest changes to the code. Follow the link to set up the cloning of the repository. Afterwards, when new versions come, you can pull the changes to your computer, [like here](https://help.github.com/articles/fetching-a-remote/).
- Download the code: At the top right of the github page, you'll see a green button written _Clone or download_.


### Preparation of leaf microCT images for semi-automatic segmentation

Before the development of the machine learning segmentation tool, we were segmenting by hand our stack (as in [Théroux-Rancourt et al. (2017)](#references)). We were drawing over a vein on one slice for example, adding that region of interest (ROI) to the ROI manager in ImageJ (using the _t_ keyboard shortcut), then moving the ROI to the new position of that vein (or drawing over), adding that ROI to the manager, and so on until we had covered that specific vein over the whole stack. We were then interpolating the ROIs, which creates a ROI for each slice, and the filling that vein a specific color. This was done for all tissues and was quite time consuming, especially for non-parallel veins.

To prepare your image for the machine, you need to prepare high quality hand segmentation of a few slices of your full stack. My experience for now has showed me that the better the hand segmentation is, the better the trained segmentation model will be.

For now, I have used between 6 to 14 hand segmented slices with equal success, bu I haven't tested what is the effect of the number of slices on different leaf types. I think 6 slices would be a very acceptable the minimum, but this minimum would depend on the venation pattern and the scan's quality for example.

#### How hand segmentation is done to create testing and training labeled slices
Start several slices away from the edges, so that you cover at least three cell layers in the palisade and at least one full cell in the spongy. Some steps in the machine learning segmentation (e.g. local thickness) do not produce good results near the beginning and the end of the stack, so it's better to avoid those. For example, on a _Vitis vinifera_ scan done at 40x, we avoided the first and last 80 slices.

Each tissue is drawn over in ImageJ using the _pencil_ or _paintbrush_ tool. It is easier than using the _polygon selection_ tool as you can easily pause and also undo changes, and you can make mistakes that won't matter in the end (see pictures below). If you have some tissues touching each other, use another color. I generally draw in black over the _gridrec_ stack, and draw in white tissues touching others, like the bundle sheath (white) touching the epidermis (black). This is what it looks like:

<p align="center">
	<img src="imgs_readme/C_I_12_Strip1_01_ImageJ_draw_over_slice.png" alt="Slice drawn over" width="600">
</p>

I follow then these steps, and you can see the output below. The order in which the ROIs are added is important for the later steps:
- I use then the _magic wand_ selection tool to select the other portion of one epidermis, then hit _t_ to add it to the ROI manager. I repeat it for the other epidermis.
- I then draw a _polygon selection_ passing through each epidermis so that it creates a polygon encompassing the whole mesophyll. This selection is added to the ROI manager and will be used to create a background for the testing/training slices.
- I move now over each vein/bundle sheath pair, selecting the bundle sheath first with the _magic wand_, adding it to the ROI manager, and repeating that for the vein. I repeat this step for each vein/bundle sheath pair.

<p align="center">
	<img src="imgs_readme/C_I_12_Strip1_02_ImageJ_draw_over_slice_w_ROIs.png" alt="Slice with ROIs" width="600">
</p>

Several ROIs are now in the ROI manager. I save all of them by selecting them all (e.g. using _ctrl+a_ in the ROI manager) and then saving them (_More... > Save_ in the ROI manager). The filename is up to up, but I recommend adding the slice number to it, which is usually the first 4 digits of a ROI in the ROI manager. It's important to keep the extension `.zip`.

Once you're done with a slice and have saved the ROI set, clear the ROI manager and repeat the above on another slice.

After having created a ROI set for each draw-over slice (i.e. test/training slices), I use a [custom ImageJ macro](ImageJ_macros/Slice%20labelling%20-%20epidermis%20and%20BS.ijm.ijm). I've created a few over time depending on which tissues I wanted to segment, all named `Slice labelling`. Ask me for which would suit you best and how to edit it. This macro loops over the ROI sets in a folder and creates a labeled stack consisting of the manually segmented tissues painted over the binary image (i.e. the image combining the thresholded gridrec and phase stacks). It only labels the tissues mentioned above, so if you want more, contact me or try it yourself.

I first open the binary stack. By binary stack, I mean the stack created by combining the thresholded _gridrec_ and _phase contrast_ images, as done in [Théroux-Rancourt et al. (2017)](#references) and like in the picture below.

<p align="center">
	<img src="imgs_readme/C_I_12_Strip1_IMGJ_GRID_PHASE_Threshold_w_menu.jpg" alt="Thresholding example">
</p>

This binary stack should be in the same folder as your ROI sets if you plan on using the macro mentioned above. The macro will fing all `.zip` file in the folder the binary stack is, open each one, clears the background outside the mesophyll, fills up the epidermises, the bundle sheaths, and the veins. Below, you see how the binary stack ends up in the segmented stack.

<p align="center">
	<img src="imgs_readme/C_I_12_Strip1_00c_binary-slice0440.png" alt="Binary slice" width="600">
	<img src="imgs_readme/C_I_12_Strip1_00d_labelled-slice0440.png" alt="Segmented slice" width="600">
</p>



Now, a new file name `labelled-stack.tif` (_Note: this typo will be corrected_) is in the folder your binary image was, and this is the stack needed for training and testing the machine learning segmentation model. A window has also opened with the names of all the `.zip` files. Copy that line to a text editor and keep only the slice numbers: you will need the sequence of slice numbers for the automated leaf segmentation.

Before moving on to the next step, make sure that your __files are named in a regular way__. For example, in `Carundinacea2004_0447_GRID-8bit.tif`, the sample name (`Carundinacea2004_0447_`) and file type (`GRID-8bit`, the _gridrec_ file for that sample) are present. This constant file naming is necessary for the leaf segmentation program to run smoothly. Also, the folder should have the same name as the sample (i.e. `Carundinacea2004_0447_` in this example).

Finally, a note about __bit depth__. Preferably, use 8-bit images for the machine learning segmentation. Files are smaller in size and it will take up less RAM. However, the program can have 16 or 32-bit images as input, as long as the right threshold value is used as an input (see next section).

### Leaf segmentation: `Leaf_Segmentation.py`
The program is currently setup to run non-interactively from the command line. I chose this in order to run multiple segmentation processes overnight. Another advantage is that it clears the memory efficiently when the program ends.

The program is run from the command line interface (`terminal` under macOS, `cmd` in Windows, whatever terminal you use under Linux). Note that under Windows, it is preferable to set the path to your python distribution, [as described here](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows).

From the terminal window, the program is called like this:

```
python /path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py filename_ PHASE GRID 'list,of,slices,in,imagej,1,2,3,4' rescale_factor threshold_rescale_factor '/path/to/your/image/directory/' nb_of_estimators
```

Real example:

```
python ~/Dropbox/_github/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py Carundinacea2004_0447_ 82 123 '83,275,321,467,603,692' 1 1 '/run/media/gtrancourt/GTR_Touro/Grasses_uCT/'
```

`python`: This just calls python 2.

`/path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py`: This should be the complete path to where the segmentation program is. If you have cloned the repository from github, replace `/path/to/this/repo/` for the path to the repository. This is also the folder in which the functions code is located (`Leaf_Segmentation_Functions.py`) and this file is called by `Leaf_Segmentation.py`. _The functions will be soon merged into the segmentation code._

`filename_`: This the filename and the name of the folder. Right now, it is setup so that the folder and the base file name are exactly the same. By base file name, I mean the first part of your naming convention, like `Carundinacea2004_0447_` which is the name of the folder and also exactly the same as in `Carundinacea2004_0447_GRID-8bit.tif`, the gridrec file name.

`PHASE` and `GRID`: These are the threshold values for the phase contract image (also called paganin reconstruction). These values are the ones taken from the threshold menu (see picture comparing gridrec and phase thresholding above), and imply in the program that all values between 0 and the value entered will be converted to white and the other values to black. In the picture below, the region in red will be the one that will be thresholded. As explained in [Théroux-Rancourt et al. (2017)](#references) and shown in a picture above, both gridrec and phase contrast thresholded images are combined together to make a binary image encompassing fine details around the cells and the bulk of the airspace. Hence only one value is needed per reconstruction type in the command line.

`'list,of,slices,in,imagej,1,2,3,4'`: This is the list of slices in ImageJ notation, i.e. with 1 being the first element. Needs to be between `''` and separated by commas.

`rescale_factor`: Default is 1 (no rescaling). Depending on the amount of RAM available, you might need to adjust this value. For stacks of smaller size, ~250 Mb, no rescaling should be necessary. Files larger than 1 Gb should be rescaled by 2. This is a downsizing integer that can be used to resize the stack in order to make the computation faster or to have a file size manageable by the program. It will resize only the _x_ and _y_ axes and so keeps more resolution in the _z_ axis. These files are used during the whole segmentation process. Note that the resulting files will be anisotropic, i.e. one voxel has different dimension in _x_, _y_, and _z_.

`threshold_rescale_factor`: Default is 1 (no rescaling). This one resizes _z_, i.e. depth or the slice number, after having resized using the `rescale_factor`. This is used in the computation of the local thickness (computing the largest size within the cells -- used in the random forest segmentation). This is particularly slow process and benefits from a smaller file, and it matters less if there is a loss of resolution in this step. Note that this image is now isotropic, i.e. voxels have same dimensions in all axes.

`'/path/to/your/image/directory/'`: Assuming all your image folder for an experiments are located in the same folder, this is the path to this folder (don't forget the `/` at the end).

__Optional input__: `nb_of_estimators`: Default is 50 when no value provided at the end of the command. The number of estimators, or trees, used in the random forest classification model. Usually between 10 and 100. Increasing the value will increase the model size (i.e. more RAM needed) and may not provide better classification.



Before running your first automated segmentation, you should look at lines 73 to 75 of `Leaf_Segmentation.py` to change the naming convention you use for _gridrec_ and _phase contrast_ stacks. I use `GRID-8bit.tif` for my _gridrec_ stacks (and I specify that they are 8 bit images), but you could use another name (e.g. `GR`). If the naming isn't right, an error message will be printed.

The program will be independent once you launch the command. It will print out some messages saying what is being done and some progress bars for the more lengthy computations. It can take several hours to segment your whole stack.


### Post-processing and leaf traits analysis
Post-processing and leaf traits analysis detailed procedure to come.

A jupyter notebook rendering of the post-processing and leaf trait analysis code, with some resulting images in the notebook, is available. This isn't the most up-to-date code, but will give you a good idea of what's happening. The leaf traits code is fairly well commented.

Please note that this code will most probably have to be fine-tuned to each experiment.


### Leaf tortuosity and airspace diffusive traits analysis: `Leaf_Tortuosity.py`
A python version of the method used by [Earles et al. (2018)](#references) has been developped and can be used from the command line. It is now stable and has been used to analysis more than 30 segmented scans in an automated command line function. There are still some glitches with certain stacks and I will troubleshoot that in the following days. Please contact me if you use this function in order to improve it. If you run into an error, please create an issue and copy the error message into it. An interactive version, probably as notebook, will be produced from this code.

Note that the code works only for hypostomatous leaves with stomata on the abaxial surface at the moment. I will implement other types of leaves as I run into them. If you have some, please contact me.

If you use this code to compute the geometric tortuosity and the path lengthening, please cite it as something like this ([full reference to the paper here](#references)):

> ... using the python version (github.com/gtrancourt/microct-leaf-traits) of [Earles et al. (2018)](#references) method...


 
#### What is being produced by the function:

Stacks as tiff files

- 3D stacks of the Tortuosity factor and Path length at the edge of the airspace (32-bit)
- 2D means in cross and longitudinal sections of the above (32-bit)
- 3D stack of the unique stoma airspace regions, i.e. the regions of the airspace colored according to the closest stoma. This is computed by computing the geodesic distance starting for each stoma and comparing with the global geodesic distance. A unique region is then defined as L<sub>geo</sub><sup>i</sup> = L<sub>geo_leaf</sub>, where _i_ is the number of the stoma.
- 3D stack of the airspace outline touching only mesophyll cell, but not touching the epidermis (i.e. the edge of the mesophyll cells).

Results as text files

- 1D profile between epidermis for the Tortuosity factor, Path length, Surface area, and Porosity. Includes also the background space outside of the epidermises.
- Surface area and pore volume of each unique stoma region that are fully bordered by othert stoma regions, i.e. that are not touching the border. Hence, these are the fully regions and avoid the chances a stoma might be just outside the stack and bias the region of influence of a stoma. Data in the unit associated with the pixel size used as input.
- Mean, median, standard deviation, variance, skewness, and value at 50% surface for Tortuosity factor and Path length. By 50% surface, I mean the position within the leaf profile where there is as much mesophyll cell surface area above and below.

***

#### How to use the code

You need a segmented stack with the stomata labelled with a color value that is different to all of the other tissue values of the segmented stack. In ImageJ, I use _red_ (value of 85 in 8-bit), which is different from all other values. The stoma are labelled as an circle or ellipse that start at the point where the stoma opens and is connected to the airspace. I usually draw the ellipse on four slices since I reduce the size of my stacks because of their very large size and the computing power needed for them (i.e. it stalls my 64GB RAM machines too much). If one stoma is not connected to the airspace, the program will stop. But if they are well labelled, then it will work out very well.

To use the code, type the line below in your terminal window:

```
python /path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Tortuosity.py sample_folder/full_filename rescale_factor pixel_size 'tissue_values' nb_cores '/path/to/your/image/directory/'
```

Real example:

```
python ~/Dropbox/_github/microCT-leaf-traits/Leaf_Tortuosity.py C_I_2_Strip1_/C_I_2_Strip1_SEGMENTED.tif 2 0.1625 default 6 '/run/media/gtrancourt/microCT_GTR_8tb/Vitis_Shade_Drought/_ML_DONE/'
```

`python`: This just calls python 2.

`/path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Tortuosity.py`: This should be the complete path to where the tortuosity program is. If you have cloned the repository from github, replace `/path/to/this/repo/` for the path to the repository (same as for the leaf segmentation code).

`sample_folder/full_filename`: This is the name of the folder in which the segmented stack is, followed by the full segmented stack's name. This will be joined to the path to your image directory. Having both the folder and the full name of the stack here allows to automatically apply the tortuosity function to files found in a base directory (see below for a usage example).

`rescale_factor`: Depending on the amount of RAM available, you might need to adjust this value. I use 2 for large stacks (> 2 Gb). Contrary to the `rescale_factor` in the `Leaf_Segmentation.py`, here the stack is resized in all three dimensions. See `Leaf_Segmentation.py` for other comments on resizing.

`pixel_size`: The length of a pixel. Can be any unit. Allows for the computation of the size related traits.

`'tissue_values'`: Either `default` for the default values I use or a string of values correspond to the pixel value, found in ImageJ, for the following tissues in this exact order (default values in parentheses): mesophyll cells (0), background (177), airspace (255), stomata (85), adaxial epidermis (30), abaxial epidermis (60), veins (147), bundle sheath (102). Repeat the vein value for the bundle sheath if the latter is not segmented, and same thing for the epidermises. In the command line, this would look like: `'0,177,255,85,30,60,147,102'` (don't forget `''`).

`nb_cores`: Optional. If not provided, will use all cores available for some more intensive computation.

`'/path/to/your/image/directory/'`: Assuming all your image folder for an experiments are located in the same folder, this is the path to this folder (don't forget the `/` at the end).



If you want to loop over your image directory, you can do so easily in a UNIX environment using the `find` command. Do do so, open your terminal and change directory up to your image folder (the `'/path/to/your/image/directory/'`). Then, you can loop over all the segmented stacks in that directory like this:

```
find -iname *SEGMENTED.tif -exec python ~/Dropbox/_github/microCT-leaf-traits/Leaf_Tortuosity.py {} 2 0.1625 default 6 '/run/media/gtrancourt/microCT_GTR_8tb/Vitis_Shade_Drought/_ML_DONE/' \;
```

Here, `find` searches for all files ending with `SEGMENTED.tif` and, for each file found pipes it to the `Leaf_Tortuosity.py` function through `{}`, which correspond to a character string with the path from the current directory up to the file found. In my case, `{}` would be replaced by `./C_I_2_Strip1_/C_I_2_Strip1_SEGMENTED.tif` for example, where `./` represent the current directory (which is already specified in the call to the function, so this isn't used).



## Changes made to the [previous version](https://github.com/mattjenkins3/3DLeafCT)
The core of the machine learning functions hasn't changed from the work done by Matt Jenkins and Mason Earles and as such they are the sole authors of the core functions such as `RFPredictCTStack` and `GenerateFL2`. Functions feeding images to be trained or segmented have been updated or changed in depth, such as how the files were saved, their bit depth, and other features that do affect the memory usage and the speed of computation.

#### Segmentation code
- Split the function and program in two files. `Leaf_Segmentation_Functions.py` is where all the functions are at. This file needs some tidying to remove unnecessary code. `Leaf_Segmentation.py` is the file used to carry out the segmentation.
- Non-interactive, on the command line [(see above)](#leaf-segmentation-leaf_segmentationpy). I preferred a non-interactive use to simplify my usage of the program and be able to run several segmentation processes over night. This also allows to flush the memory every time the program ends, which is essential with the file size used here.
- Runs the segmentation on a downsized stack as my microCT scans were too large to be run in a reasonable amount of time (i.e. < 6 hours) and with a usable amount of RAM (i.e. not capping 64 Gb of RAM and 64 Gb of swap memory). My scans are 3-5 Gb in size, so the downscale size is ~1 Gb, which works out fine. The resizing factor can be changed to `1` (no resize) by changing the `rescale_factor` value.
- Using now the `resize` instead of `rescale` function in order to resize axes 1 and 2 (x and y in ImageJ). The `rescale` function resized axes 0 and 1 (z and x). It made more sense to me to resize x and y.
- Changed the resize function to use a nearest neighbor interpolation (`order=0` in the function; the previous version probably forgot to set that). This prevents the introduction of new values in the predicted stacks, both in the model and final full stack prediction.
- Resized the local thickness file input a second time, this time on axis 0 (z in ImageJ), i.e. the stack used to compute local thickness is now isotropic (all edges of one voxel have the same dimension), which is not the case with the resized images above. This is the binary image generated in function `Threshold_GridPhase_invert_down`. Saves some time in the end, and avoids crashing your computer, even when it has 64 Gb of RAM.
- Added a trimming function so that the original images can be divided by a divider of choice (mainly 2 in my case). This removes one or two pixel-wide rows or columns at the outermost edge (bottom or right on the image). This allows to reuse the same image afterward and to compare the downsized full stack predictions.
- Now automatically saves the random forest model using the `joblib` package. It's fast and a great compression ratio is achieved, so there's no use not to keep it as we can extract information on the model afterwards.
- Saves the local thickness file as 8 bits integer. 256 values are more than enough in my case, but this could easily be switched to 16 bits. Local thickness is only computed as pixel values, so it doesn't matter if there's a lost in resolution.
- Added several steps in the code where, if a file is already present, it will load it instead of generating it. This is useful if the program crashed at one point (i.e. the local thickness file was generated after 3 hours of processing, but something happened to the computer and it crashes in the middle of the model training).
- Randomized the training and testing slices order. I found it preferable to mix up the order. I currently uses half for training, half for testing.
- Uses the ImageJ slice indexing as an input. Avoids having to subtract one to all your slice numbers. I generate my labeled stack of training and testing slices using [this ImageJ macro](https://github.com/gtrancourt/imagej_macros/blob/master/macros/Batch%20slice%20labelled%20with%20epidermis%20over%20multiple%20RoiSets.ijm).
- I've removed the computation of the performance metric, beside mentioning the accuracy of the model. Performance metric could be computed later using the saved random forest model (in the `joblib` format).

#### Post-processing and leaf traits analysis code
I am not using the post-processing that was in the previous code as it didn't improve my own scans. Rather, I select the specific tissues (e.g. veins, epidermis) find the largest structures and paste those structures on a binary image of the combined gridrec and phase contrast images. The leaf trait analysis was still under development in the previous versions of the code. I have written it from scratch based on my needs. It now computes the traits below and exports the data to a CSV file.

- Thicknesses: Leaf, Epidermis (adaxial and abaxial separately), Mesophyll (everything but the epidermis)
- Volumes: Leaf, Mesophyll (everything but the epidermis), Vein, Mesophyll cells, Airspace, Epidermis (adaxial and abaxial separately)
- Surface area: Airspace

## References
__Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR (2018)__ [Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact Mesophyll Conductance.](http://www.plantphysiol.org/content/178/1/148) Plant Physiol 178: 148-162.

__Théroux-Rancourt G, Earles JM, Gilbert ME, Zwieniecki MA, Boyce CK, McElrone AJ, Brodersen CR (2017)__ [The bias of a two-dimensional view: comparing two-dimensional and three-dimensional mesophyll surface area estimates using noninvasive imaging.](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.14687) New Phytol, 215: 1609-1622.


## Contributors

Active
-   [Guillaume Théroux-Rancourt](https://github.com/gtrancourt)

Previous version of leaf segmentation code. These contributors wrote the machine learning code used for segmentation, which hasn't been touched in this version.
-   [Mason Earles](https://github.com/masonearles)
-   [Matt Jenkins](https://github.com/mattjenkins3)

## Comments and contributions

I welcome comments, criticisms, and especially contributions! GitHub
issues are the preferred way to report bugs, ask questions, or request
new features. You can submit issues here:

<https://github.com/gtrancourt/microCT-leaf-traits/issues>

## Meta

-   Please [report any issues or
    bugs](https://github.com/gtrancourt/microCT-leaf-traits/issues).
-   License: MIT
<!-- -   Please note that this project is released with a [Contributor Code
    of Conduct](CONDUCT.md). By participating in this project you agree
    to abide by its terms. -->


## Change log
##### 2019-03-14
- Tortuosity code now accepts specific color values for each tissue.
- Save the mesophyll edge as a stack (i.e. the airspace minus the surface touching the epidermis).

##### 2019-03-11 - Tortuosity code only
- Tortuosity code is now fully fonctionnal and can be run from the command line.
- Found an error in how the epidermis edge was computed, which caused problem with how the segmentation causes un-smoothed epidermis. Fixed it by drawing a larger epidermis edge (3 pixels length) and purifying the resulting stack to get only one epidermis edge, hence removing the smallish un-connected epidermis elsewhere. This caused a problem when both epidermis have the same label value.
- Removes the values that are direct neighbours to an epidermis when computing statistics for tortuosity and path length.

##### 2019-03-07
- Removed 'verbosity' of the random forest classifying steps. Makes for fewer on-screen outputs and a nicer progress bar when doing the full stack segmentation.
- Added the printing on the slice number used in training and testing.
- Added the number of estimators as an optional input at the end of the command line call of `Leaf_Segmentation.py`.
- Some auto-formatted coding tweaks in `Leaf_Segmentation.py`.
- Changed the images in `README.md` to a _Vitis vinifera_ 40x scan.
