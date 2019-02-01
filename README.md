# Tools to process and analyze plant leaf microCT scans

X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input. By segmentation we mean the indentification of specific tissues within the leaves as single values within an image file.

We also provide further tools to process the segmented images, to extract leaf anatomical traits commonly measured, such as in [Théroux-Rancourt et al. (2017)](#references), or to compute airspace tortuosity and related airspace diffusion traits from [Earles et al. (2018)](#references).

<!-- ![alt text][logo]

[logo]: https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/imgs_readme/leaf1.png "translucent epidermis with veins" -->

This repository combines the most up-to-date code for the segmentation and leaf traits analysis. The leaf segmentation project has been initiated by [Mason Earles](https://github.com/masonearles/3DLeafCT) and expanded by [Matt Jenkins](https://github.com/mattjenkins3/3DLeafCT). The current version has been edited by [Guillaume Théroux-Rancourt](). All leaf traits analysis were contributed by Guillaume Théroux-Rancourt.


### Go directly to the procedure for specific tools
- [(Semi-)Automated leaf segmentation](#leaf-segmentation-leaf_segmentationpy)
- [Leaf traits analysis (and segmentation post-processing)](#post-processing-and-leaf-traits-analysis)
- [Leaf tortuosity and airspace diffusive traits analysis](#leaf-tortuosity-and-airspace-diffusive-traits-analysis)


## Requirements
- __python 2__: If you are new to python, a nice and convenient way to install python is through [anaconda](https://www.anaconda.com/download/). The code hre use python 2.7, so be sure to install this version. The anaconda navigator makes it easy to install packages, which can also be installed through command line in your terminal by typing `conda install name_of_package`. Further, the anaconda navigator allows you to open specific applications to write and run python code. Spyder is a good choice for scientific computation.
- __RAM:__ Processing a 5 Gb 8-bit multi-sliced tiff file can peak up to 60 GB of RAM and use up to 30-40 Gb of swap memory (memory written on disk) on Linux (and takes about 3-5 hours to complete). Processing a 250 Mb 8-bit file is of course a lot faster (30-90 minutes) but still takes about 10 Gb of RAM. The programm is memory savvy and this needs to be addressed in future versions in order to tweak the program.


## Procedure

### Download the code to your computer

There are two ways to copy the code to your computer.

- [Clone the repository](https://help.github.com/articles/cloning-a-repository/): Using `git` will allow you to conveniently download the latest changes to the code. Follow the link to set up the cloning of the repository. Afterwards, when new versions come, you can pull the changes to your computer, [like here](https://help.github.com/articles/fetching-a-remote/).
- Download the code: At the top right of the github page, you'll see a green button written _Clone or download_.


### Preparation of leaf microCT images for semi-automatic segmentation
A more detailed explanation with images will come.

Briefly, I draw over a few slices, the number of which should be determined for each stack based on quality of the image, venation pattern and quantity, etc. After having created a ROI set for each draw-over slice (i.e. test/training slices), I use a [custom ImageJ macro](ImageJ_macros/Batch%20slice%20labelled%20with%20epidermis%20over%20multiple%20RoiSets.ijm). I've created a few over time depending on which tissues I wanted to segment, all named `Batch slice labelled...`. Ask me for which would suit you best and how to edit it. This macro loops over the ROI sets in a folder and creates a labelled stack consisting of the manually segmented tissues painted over the binary image (i.e. the image combining the thresholded gridrec and phase stacks).

### Leaf segmentation: `Leaf_Segmentation.py`
The program is currently setup to run non-interactively from the command line. I chose this in order to run multiple segmentations overnight. Another advantage is that it clears the memory efficiently when the program ends. I do need to give a better name!

Under a Unix or Linux system, the program is called like this:

```
python /path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py filename_ PHASE GRID 'list,of,slices,in,imagej,1,2,3,4' rescale_factor threshold_rescale_factor '/path/to/your/image/directory/'
```

Real example:

```
python ~/Dropbox/_github/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py Carundinacea2004_0447_ 82 123 '83,275,321,467,603,692' 1 1 '/run/media/gtrancourt/GTR_Touro/Grasses_uCT/'
```

`python`: This just calls python 2.

`/path/to/this/repo/3DLeafCT/ML_microCT/src/Leaf_Segmentation.py`: This should be the complete path to where the segmentation program is. If you have cloned the repository from github, replace `/path/to/this/repo/` for the path to the repository. This is also the folder in which the functions code is located (`Leaf_Segmentation_Functions.py`) and this file is called by `Leaf_Segmentation.py`. _The functions will be soon merged into the segmentation code._

`filename_`: This the filename and the name of the folder. Right now, it is setup so that the folder and the base file name are exactly the same. By base file name, I mean the first part of your naming convention, like `Carundinacea2004_0447_` which is the name of the folder and also exactly the same as in `Carundinacea2004_0447_GRID-8bit.tif`, the gridrec file name.

`PHASE` and `GRID`: These are the threshold values for the phase contract image (also called paganin reconstruction). Only one value needed.

`'list,of,slices,in,imagej,1,2,3,4'`: This is the list of slices in ImageJ notation, i.e. with 1 being the first element. Needs to be between `''` and separated by commas.

`rescale_factor`: This is a downsizing integer that can be used to resize the stack in order to make the computation faster or to have a file size manageable by the program. It will resize only the _x_ and _y_ axes and so keeps more resolution in the _z_ axis. These files are used during the whole segmentation process. Note that the resulting files will be anisotropic, i.e. one voxel has different dimension in _x_, _y_, and _z_.

`threshold_rescale_factor`: This one resizes _z_, i.e. depth or the slice number, after having resized using the `rescale_factor`. This is used in the computation of the local thickness (computing the largest size within the cells -- used in the random forst segmentation). This is particularly slow process and benefits from a smaller file, and it matters less if there is a loss of resolution in this step. Note that this image is now isotropic, i.e. voxels have same dimensions in all axes.

`'/path/to/your/image/directory/'`: Assuming all your image folder for an experiments are located in the same folder, this is the path to this folder (don't forget the `/` at the end).


### Post-processing and leaf traits analysis
Post-processing and leaf traits analysis detailled procedure to come.

A jupyter notebook rendering of the post-processing and leaf trait analysis code, with some resulting images in the notebook, is available. This isn't the most up-to-date code, but will give you a good idea of what's happening. The leaf traits code is fairly well commented.

Please note that this code will most probably have to be fine-tuned to each experiment.


### Leaf tortuosity and airspace diffusive traits analysis
A python version of the method used by [Earles et al. (2018)](#references) is under active development. The code is working properly, but still lacks proper analysis of the tortuosity factor, saving capabilities.



## Changes made to the [previous version](https://github.com/mattjenkins3/3DLeafCT)
The core of the machine learning function hasn't changed from the previous work of Matt Jenkins and Mason Earles. Functions feeding images to be trained or segmented have been changed in depth, such as how the files were saved, their bit depth, and other features that do affect the memory usage and the speed of computation.

#### Segmentation code
- Split the function and program in two files. `Leaf_Segmentation_Functions.py` is where all the functions are at. This file needs some tidying to remove unnecessary code. `Leaf_Segmentation.py` is the file used to carry out the segmentation. It is used non-interactively, on the command line (see below). I prefered a non-interactive use to simplify my usage of the program and be able to run several segmentations over night. This allows to flush the memory everytime the programm ends, which is essential with the file size used here.
- Runs the segmentation on a downsized stack as my microCT scans were too large to be run in a reasonnable amount of time (i.e. < 6 hours) and with a usable amount of RAM (i.e. not capping 64 Gb of RAM and 64 Gb of swap memory). My scans are 3-5 Gb in size, so the downscale size is ~1 Gb, which works out fine. The resizing factor can be changed to `1` (no resize) by changing the `rescale_factor` value.
- Using now the `resize` instead of `rescale` function in order to resize axes 1 and 2 (x and y in ImageJ). The `rescale` function resized axes 0 and 1 (z and x). It made more sense to me to resize x and y.
- Changed the resize function to use a nearest neighbour interpolation (`order=0` in the function). This prevents the introducion of new values in the predicted stacks, both in the model and final full stack prediction.
- Resized the local thickness file input a second time, this time on axis 0 (z in ImageJ), i.e. the stack used to compute local thickness is now isotropic (all edges of one voxel have the same dimension), which is not the case with the resized images above. This is the binary image generated in function `Threshold_GridPhase_invert_down`. Saves some time in the end, and avoids crashing your computer, even when it has 64 Gb of RAM.
- Added a trimming function so that the original images can be divided by a divider of choice (mainly 2 in my case). This removes one or two pixel-wide rows or columns at the outermost edge (bottom or right on the image). This allows to reuse the same image afterward and to compare the downsized full stack predictions.
- Now automatically saves the random forest model using the `joblib` package. It's fast and a great compression ratio is achieved, so there's no use not to keep it as we can extract information on the model afterwards.
- Saves the local thickness file as 8 bits integer. 256 values are more than enough in my case, but this could easily be switched to 16 bits. Local thickness is only computed as pixel values, so it doesn't matter if there's a lost in resolution.
- Added several steps in the code where, if a file is already present, it will load it instead of generating it. This is useful if the program crashed at one point (i.e. the local thickness file was generated after 3 hours of processing, but something happened to the computer and it crashes in the middle of the model training).
- Randomized the training and testing slices order. I found it preferable to mix up the order. I currently uses half for training, half for testing.
- Uses the ImageJ slice indexing as an input. Avoids having to substract one to all your slice numbers. I generate my labelled stack of training and testing slices using [this ImageJ macro](https://github.com/gtrancourt/imagej_macros/blob/master/macros/Batch%20slice%20labelled%20with%20epidermis%20over%20multiple%20RoiSets.ijm).
- I've removed the computation of the performance metric, beside mentionning the accuracy of the model. Performance metric could be computed later using the saved random forest model (in the `joblib` format).

#### Leaf traits analysis code
This analysis was a work in progress in the previous versions of the code. I have written it from scratch based on my needs. It now computes the following and export the data as a CSV file.

- Thicknesses: Leaf, Epidermis (adaxial and abaxial separately), Mesophyll (everything but the epidermis)
- Volumes: Leaf, Mesophyll (everything but the epidermis), Vein, Mesophyll cells, Airspace, Epidermis (adaxial and abaxial separately)
- Surface area: Airspace

## References
__Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR (2018)__ [Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact Mesophyll Conductance.](http://www.plantphysiol.org/content/178/1/148) Plant Physiol 178: 148–162.

__Théroux‐Rancourt G, Earles JM, Gilbert ME, Zwieniecki MA, Boyce CK, McElrone AJ, Brodersen CR (2017)__ [The bias of a two‐dimensional view: comparing two‐dimensional and three‐dimensional mesophyll surface area estimates using noninvasive imaging.](https://nph.onlinelibrary.wiley.com/doi/full/10.1111/nph.14687) New Phytol, 215: 1609-1622.


## Contributors

Active
-   [Guillaume Théroux-Rancourt](https://github.com/gtrancourt)

Previous version of leaf segmentation code
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
