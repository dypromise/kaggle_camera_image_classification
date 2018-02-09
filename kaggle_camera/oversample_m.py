def _oversample(images, crop_dims, crops_num):

    im_shape = np.array(images[0].shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates

    delta_h = int(im_shape[0]-crop_dims[0])/crops_num
    delta_w = int(im_shape[1]-crop_dims[1])/crops_num
    h_indices = [x*delta_h for x in range(int(crops_num))]
    w_indices = [x*delta_w for x in range(int(crops_num))]

    crops_ix = np.empty((crops_num**2, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    
    crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((crops_num**2*2 * len(images), crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    for im in images:
        for crop in crops_ix:
            crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
            ix += 1
        crops[ix- crops_num**2:ix] = crops[ix- crops_num**2:ix, :, ::-1, :]  # flip for mirrors
    return crops