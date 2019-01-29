// Automatically segment microCT leaf cross sections for the use of the
// 3DLeafCT machine learning algorithm (https://github.com/masonearles/3DLeafCT)
//
// This macro is to segment over on a binary image of the airspace.
//
// ROIs (Regions of interest in ImageJ) sets (zip files containing ROIs)
// should be arranged in the RoiManager in the order below:
// - 1st and 2nd elements: Epidermis
// - 3rd element: Everything between the two epidermis
// - 4th until the end: Veins/Vasculature/Other features
//
// Files should be arranged sequentially in you folder, but this is less
// if you specify the right sequence in the 3DLeafCT programm.
//
// Author: Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
// Created on 16.02.2018
// Last modification on 18.10.2018
//
// TO DO:
// - Have all the ROIs in the same RoiSet and detect the slice value
//   (instead of iterating over multiple zip files)
// - Create a super-macro that let's you chose what file organisation
//   has been used.


// Get information out of the opened image
fullIMG = getTitle();
getPixelSize(unit, pw, ph, pd);

// Get the ROIs
// Create RoiSets with all the zip file names for individual slices
dir = getDirectory("image");
list = getFileList(dir);

Array.print(list);

RoiSets = newArray(1);

for (k=0; k<list.length; k++) {
	if (endsWith(list[k], ".zip"))
		RoiSets = Array.concat(RoiSets, list[k]);
}

RoiSets = Array.slice(RoiSets,1);

Array.show(RoiSets);

for (i=0; i<RoiSets.length; i++) {
	selectWindow(fullIMG);
	roiManager("open", dir+RoiSets[i]);
	run("Colors...", "foreground=orange background=yellow selection=yellow");

	//Count the number of ROIs. Will skip the first 2 below
	nROIs = roiManager("count");

	// Select all the veins and fill them.
	for (j=3; j<nROIs; j++) {
		selectWindow(fullIMG);
		roiManager("Select", j);
		run("Fill", "slice");
	}

	run("Colors...", "foreground=green background=yellow selection=yellow");
	roiManager("Select", 2); //Select the 3rd ROI = Mesophyll
	run("Clear Outside", "slice");

	roiManager("Select", 0); //Select the 1st ROI = Epidermis
	run("Fill", "slice");
	roiManager("Select", 1); //Select the 2nd ROI = Epidermis
	run("Fill", "slice");



	// Copy each labelled slice to a new file
	// Create that image stack the first time
	run("Select All");
	run("Copy");

	if (i == 0) {
		run("Internal Clipboard");
	} else {
		selectWindow("Clipboard");
		run("Add Slice");
		run("Paste");
	}

	roiManager("reset")
}

// Set the scale and save the file
selectWindow("Clipboard");
run("Set Scale...", "distance=1 known="+pd+" unit="+unit);
saveAs("Tiff", dir + "labelled-stack")
