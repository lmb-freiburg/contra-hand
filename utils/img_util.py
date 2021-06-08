import numpy as np
import cv2


def downsample(img, target_size, K=None, borderValue=0.0):
    """
        img, HxWxC image
        target_size, shape in (height, width)
        K, camera intrinsic matrix
    """
    f_y = float(target_size[0]) / img.shape[0]
    f_x = float(target_size[1]) / img.shape[1]

    # how to account for crop in intrinsics
    M = np.array([[f_x, 0.0, 0.0],
                  [0.0, f_y, 0.0],
                  [0.0, 0.0, 1.0]])

    img_c = cv2.warpAffine(img, M[:2, :],
                           (target_size[0], target_size[1]),
                           borderValue=borderValue)
    if K is None:
        return img_c
    K_c = np.matmul(M, K)
    return img_c, K_c


def random_crop(img, K=None,
                f_trans=0.05,  # percent of the image size
                f_scale_min=0.8, f_scale_max=1.0,  # percent of the
                target_size=128, borderValue=0.0):
    center = np.array([img.shape[1], img.shape[0]], dtype=np.float32) / 2.0
    size = np.array([img.shape[1], img.shape[0]], dtype=np.float32)

    # random translation
    f = np.random.rand(2, ) * 2 * f_trans - f_trans
    trans_uv = f*size

    # random scaling
    f = np.random.rand() * (f_scale_max - f_scale_min) + f_scale_min

    trans_uv -= center*(1.0-f)  # translation of the image center due to scaling

    f *= 224.0 / target_size

    # how to account for crop in intrinsics
    M = np.array([[1.0 / f, 0.0, trans_uv[0] / f],
                  [0.0, 1.0 / f, trans_uv[1] / f],
                  [0.0, 0.0, 1.0]])

    img_c = cv2.warpAffine(img, M[:2, :], (target_size, target_size), borderValue=borderValue)
    if K is None:
        return img_c
    K_c = np.matmul(M, K)
    return img_c, K_c


def crop(img, center, size, K=None, target_size=128, borderValue=0.0, scale_values=False):
    size = np.max(size)*np.ones_like(size)
    size = (size/2.0).round().astype(np.int32) # this cant be a float

    # create crop image
    borderValue = np.array(borderValue).astype(img.dtype)
    img_crop = borderValue * np.ones((2*size[0], 2*size[1], img.shape[2]),
                                     dtype=img.dtype)  # after mean subtraction 127.5 will be zero

    # figure out where we would like to crop (can exceed image dimensions)
    start_t = (center - size).round().astype(np.int32)
    end_t = start_t + 2*size

    # check if there is actually anything to be cropped (sometimes crop is completely out of the image).
    do_crop = True

    # sanity check the crop values (sometime the crop is completely outside the image)
    if np.any(np.logical_or(end_t < 0, start_t > np.array(img.shape[:2]) - 1)):
        print('WARNING: Crop is completely outside image bounds!', center, img.shape)
        do_crop = False

    # check image boundaries: Where can we crop?
    start = np.maximum(start_t, 0)
    end = np.minimum(end_t, np.array(img.shape[:2]) - 1)

    # check discrepancy
    crop_start = start - start_t
    crop_end = 2*size - (end_t - end)

    if do_crop:
        img_crop[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], :] = img[start[0]:end[0], start[1]:end[1], :]
    offset = start - crop_start

    scale = (end - start) / np.array([target_size, target_size], dtype=np.float32)
    img_crop = cv2.resize(img_crop, (target_size, target_size))

    if scale_values:
        # makes sense if the image is a flow
        img_crop[:, :, 0] /= scale[1]
        img_crop[:, :, 1] /= scale[0]

    if K is not None:
        # how to account for crop in intrinsics
        A = np.array([[1.0/scale[1], 0.0, -offset[1]/scale[1]],
                      [0.0, 1.0/scale[0], -offset[0]/scale[0]],
                      [0.0, 0.0, 1.0]])
        return img_crop, np.matmul(A, K.copy())
    return img_crop

