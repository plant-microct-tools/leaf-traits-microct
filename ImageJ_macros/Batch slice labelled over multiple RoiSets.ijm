// Automatically segment microCT leaf cross sections for the use of the
// 3DLeafCT machine learning algorithm (https://github.com/masonearles/3DLeafCT)
//
// This macro is to segment over on a binary image of the airspace.
//
// ROIs (Regions of interest in ImageJ) sets (zip files containing ROIs)
// should be arranged in the RoiManager in the order below:
// - 1st element: Mesophyll only (i.e. everything expect the epidermis)
// - 2nd element: Whole leaf
// - 3rd until the end: Veins/Vasculature
//
// Files should be arranged sequentially in you folder, but this is less
// if you specify the right sequence in the 3DLeafCT programm.
//
// Author: Guillaume Théroux-Rancourt (guillaume.theroux-rancourt@boku.ac.at)
// Created on 16.02.2018
// Last modification on 17.09.2018
//
// TO DO:
// - Have all the ROIs in the same RoiSet and detect the slice value
//   (instead of iterating over multiple zip files)


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
	run("Colors...", "foreground=green background=orange selection=yellow");

	//Count the number of ROIs. Will skip the first 2 below
	nROIs = roiManager("count");

	for (j=2; j<nROIs; j++) {
		selectWindow(fullIMG);
		roiManager("Select", j);
		run("Fill", "slice");
	}

	roiManager("Select", 0); //Select the 1st ROI = Mesophyll
	run("Clear Outside", "slice");
	run("Colors...", "foreground=green background=yellow selection=yellow");
	roiManager("Select", 1); //Select the 2nd ROI = Mesophyll + Epidermis
	run("Clear Outside", "slice");


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
