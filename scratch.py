def tissue_cleanup_and_analysis(stack, tissue_name, tissue_color, full_tissue, surface_area, SA_step=2, volume_threshold, px_dimension, units):
    # stack: numpy array - fullstack reconstruction
    # tissue_name: string - name of the tissue for output only
    # full_tissue: boolean - is the tissue whole and should create a single output tissue,
    #                        or is it split and should be split into two tissues (e.g. epidermis)
    # surface_area: boolean - should surface area be computed or not
    # SA_step: integer - step size for the marching cube algorithm. GTR's tests showed that 2 is closer to the
    #                    mathemical volume of a sphere image than 1 (i.e. some smoothing out is needed)
    # volume_threshold: integer - below what pixel volume should unique volumes in that
    #                             tissue be not considered part of the tissue.
    # px_dimension: tuple (?) - the dimensions of a pixels, i.e. depth, width, height
    # units: string - units for the pixel dimensions

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
        # # Find the values of each volumes: assumes adaxial volumes is at the top of the image

        # GTR NOTE: This is a loose thread - we should find a way not to check on the 100th slices and 100th column from the top.
        #           There should be a better way to find what is the value at the top and at the bottom
        first_volume_value = unique_volumes_volumes[100, :, 100][(
                 unique_volumes_volumes[100, :, 100] != 0).argmax()]
        second_volume_value = int(np.arange(start=1, stop=3)[
                                           np.arange(start=1, stop=3) != first_volume_value])

        # MOVE OUTSIDE OF FUNCTION INTO A NEW FUNCTION ?
        # Compute volume
        first_volume_volume = volumes_area[first_volume_value - 1] * np.prod(px_dimension)
        second_volume_volume = volumes_area[second_volume_value - 1] * np.prod(px_dimension)

        # Thickness return a 2D array, i.e. the thcikness of each column
        first_volume_thickness = np.sum((unique_volumes_volumes == first_volume_value), axis=1) * np.prod(px_dimension[1:])
        second_volume_thickness = np.sum((unique_volumes_volumes == second_volume_value), axis=1) * np.prod(px_dimension[1:])
        del props_of_unique_volumes
        gc.collect()

        computed_volume = (first_volume_volume, second_volume_volume)
        computed_thickness = (first_volume_thickness, second_volume_thickness)

        # Print the results
        print('volume of first '+tissue_name+': ', first_volume_volume)
        print('volume of first ' + tissue_name + ': ', second_volume_volume)
        print('thickness of first '+tissue_name+': ', np.median(first_volume_thickness))
        print('thickness of first ' + tissue_name + ': ', np.median(second_volume_thickness))

    else:
        # Remove volumes below a threshold
        large_volumes_ids = volumes_label[volumes_area > volume_threshold]
        # Find the largest volumes
        tissue_cleaned_stack = np.in1d(unique_volumes, large_volumes_ids).reshape(stack.shape)

        computed_volume = np.sum(tissue_cleaned_stack) * np.prod(px_dimension)
        # NOT TESTED BELOW
        computed_thickness = np.sum(tissue_cleaned_stack, axis=1) * np.prod(px_dimension[1:])

        del unique_volumes
        del props_of_unique_veins
        gc.collect()

        print('volume of ' + tissue_name + ': ', computed_volume)
        print('thickness of '+tissue_name+': ', np.median(computed_thickness))

    if surface_area:
        print("")
        print('### Computing surface area')
        print('### This may take a while and freeze your computer')
        vert_faces = marching_cubes_lewiner(
            tissue_cleaned_stack, 0, allow_degenerate=False, step_size=SA_step, spacing=px_dimension)
        computed_SA = mesh_surface_area(vert_faces[0], vert_faces[1])
        print(('surface area of '+tissue_name+': '+str(ias_SA)+' '+units+'**2'))
    else:
        computed_SA = -1

    return tissue_name, tissue_color, tissue_cleaned_stack, computed_volume, computed_thickness, computed_SA
