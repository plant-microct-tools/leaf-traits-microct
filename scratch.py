def tissue_cleanup(stack, tissue_name, tissue_color, full_tissue, volume_threshold):
    # Label all of the individual regions/volume in the tissue
    unique_volumes = label(stack == tissue_color, connectivity=1)
    props_of_unique_volumes = regionprops(unique_volumes)
    
    # Find the size and properties of the unique regions
    volumes_area = np.zeros(len(props_of_unique_volumes))
    volumes_label = np.zeros(len(props_of_unique_volumes))
    volumes_centroid = np.zeros([len(props_of_unique_volumes), 3])
    for regions in np.arange(len(props_of_unique_volumes)):
        volumes_area[regions] = props_of_unique_volumes[regions].area
        volumes_label[regions] = props_of_unique_volumes[regions].label
        volumes_centroid[regions] = props_of_unique_volumes[regions].centroid
    ordered_volumes = np.argsort(volumes_area)

    # This case is when the desired tissue is split into two in the stack
    if full_tissue == False:
        # Find the two largest volumes - E.g. the two epidermis
        print('The two largest values below should be in the same order of magnitude')
        print((volumes_area[ordered_volumes[-4:]]))
        if volumes_area[ordered_volumes[-1]] > (10 * volumes_area[ordered_volumes[-2]]):
            print('#########################################')
            print('#########################################')
            print('ERROR: Both volumes are still connected!')
            print('' + sample_name)
            print('#########################################')
            print('#########################################')
            assert False

        print("")
        print('The center of the volumes should be more or less the same on the')
        print('1st and 3rd columns for the two largest values.')
        print((volumes_centroid[ordered_volumes[-2:]]))
        print("")

        two_largest_volumes = (unique_volumes_volumes
                                 == ordered_volumes[-1] + 1) | (unique_volumes_volumes == ordered_volumes[-2] + 1)

        # Check if it's correct
        # io.imsave(filepath + folder_name + 'test_volumes.tif',
        #          img_as_ubyte(two_largest_volumes))
        # io.imshow(two_largest_volumes[100])

        # Get the values again: makes it cleaner
        unique_volumes_volumes = label(two_largest_volumes, connectivity=1)
        props_of_unique_volumes = regionprops(unique_volumes_volumes)
        volumes_area = np.zeros(len(props_of_unique_volumes))
        volumes_label = np.zeros(len(props_of_unique_volumes))
        volumes_centroid = np.zeros([len(props_of_unique_volumes), 3])
        for regions in np.arange(len(props_of_unique_volumes)):
            volumes_area[regions] = props_of_unique_volumes[regions].area
            volumes_label[regions] = props_of_unique_volumes[regions].label
            volumes_centroid[regions] = props_of_unique_volumes[regions].centroid

        ## io.imshow(unique_volumes_volumes[100])

        # Transform the array to 8-bit: no need for the extra precision as there are only 3 values
        tissue_cleaned_stack = np.array(unique_volumes_volumes, dtype='uint8')


        # THIS WILL HAVE TO MOVE OUT AND/OR CREATE A VECTOR WITH THE VALUES FOR BOTH TISSUES
        # # Find the fvalues of each volumes: assumes adaxial volumes is at the top of the image
        # adaxial_volumes_value = unique_volumes_volumes[100, :, 100][(
        #         unique_volumes_volumes[100, :, 100] != 0).argmax()]
        # abaxial_volumes_value = int(np.arange(start=1, stop=3)[
        #                                   np.arange(start=1, stop=3) != adaxial_volumes_value])

        # MOVE OUTSIDE OF FUNCTION INTO A NEW FUNCTION
        # # Compute volume
        # volumes_adaxial_volume = volumes_area[adaxial_volumes_value - 1] * (px_edge * (px_edge * 2) ** 2)
        # volumes_abaxial_volume = volumes_area[abaxial_volumes_value - 1] * (px_edge * (px_edge * 2) ** 2)
        #
        # # Tichkness return a 2D array, i.e. the thcikness of each column
        # volumes_abaxial_thickness = np.sum(
        #     (unique_volumes_volumes == abaxial_volumes_value), axis=1) * (px_edge * 2)
        # volumes_adaxial_thickness = np.sum(
        #     (unique_volumes_volumes == adaxial_volumes_value), axis=1) * (px_edge * 2)
        del props_of_unique_volumes
        gc.collect()

    else:
        # Remove volumes below a threshold
        large_volumes_ids = volumes_label[volumes_area > volume_threshold]
        # Find the largest volumes
        tissue_cleaned_stack = np.in1d(unique_volumes, large_volumes_ids).reshape(stack.shape)

        del unique_volumes
        del props_of_unique_veins
        gc.collect()

    return tissue_name, tissue_color, tissue_cleaned_stack
