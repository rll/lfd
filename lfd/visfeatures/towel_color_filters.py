from rapprentice import berkeley_pr2, clouds
from scipy import spatial
import IPython as ipy
import skimage.morphology as skim
import cv2, numpy as np
import cloudprocpy

DEBUG_PLOTS = True

def extract_nongreen(rgb, depth, T_w_k = None, returnOnlyXyz=True, mask_vals = -0.25):
    """
    extract non-green points and downsample
    keep RGB information in the returned data
    """

    right_mask_val = None
    bottom_mask_val = None
    if type(mask_vals) != tuple and type(mask_vals) != list:
        #y_mask_val = mask_vals
        y_mask_val = -0.35
    else:
        y_mask_val = mask_vals[0]

        if len(mask_vals) > 1:
            right_mask_val = mask_vals[1]
            if len(mask_vals) > 2:
                bottom_mask_val = mask_vals[2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    h_mask = (h<35) | (h>100)
    s_mask = (s >= 0)  # Ranges from 0 to 255
    v_mask = (v > 90)  # Ranges from 0 to 255
    nongreen_mask = h_mask & s_mask & v_mask

    valid_mask = depth > 0

    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    if T_w_k is None:
        xyz_w = xyz_k.copy()
    else:
        xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

    z = xyz_w[:,:,2]
    z0 = xyz_k[:,:,2]

    #height_mask = xyz_k[:,:,2] < DIST_THRESHOLD # TODO pass in parameter
    #ipy.embed()
    #x_mask = xyz_w[:,:,0] < 0.5
    if bottom_mask_val is not None:
        x_mask = xyz_w[:,:,0] > bottom_mask_val  # bottom_mask_val
    if right_mask_val is not None:
        y2_mask = xyz_w[:,:,1] > right_mask_val # To cut off gripper at right of image

    y_mask = xyz_k[:,:,1] > y_mask_val  # To cut off papers at top of image
    #x2_mask = xyz_w[:,:,1] < 1.0

    good_mask = nongreen_mask & valid_mask & y_mask
    if bottom_mask_val is not None:
        good_mask = good_mask & x_mask
    if right_mask_val is not None:
        good_mask = good_mask & y2_mask
    good_mask = skim.remove_small_objects(good_mask,min_size=64)

    if DEBUG_PLOTS:
        #cv2.imshow("z0",z0/z0.max())
        #cv2.imshow("z",z/z.max())
        cv2.imshow("hue", h_mask.astype('uint8')*255)
        cv2.imshow("sat", s_mask.astype('uint8')*255)
        cv2.imshow("val", v_mask.astype('uint8')*255)
        cv2.imshow("valid_mask", valid_mask.astype('uint8')*255)
        #cv2.imshow("height", height_mask.astype('uint8')*255)
        if bottom_mask_val is not None:
            cv2.imshow('xmask', x_mask.astype('uint8')*255)
        cv2.imshow('ymask', y_mask.astype('uint8')*255)
        #cv2.imshow('x2mask', x2_mask.astype('uint8')*255)
        if right_mask_val is not None:
            cv2.imshow('y2mask', y2_mask.astype('uint8')*255)
        cv2.imshow("final",good_mask.astype('uint8')*255)
        cv2.imshow("rgb", rgb)
        cv2.waitKey()

    good_xyz = xyz_w[good_mask]
    good_rgb = rgb[good_mask]

    #downsampled_xyz = clouds.downsample(good_xyz, DS_SIZE)

    # To each downsampled point, assign color of closest original point
    #tree = spatial.KDTree(good_xyz)
    #[_, i] = tree.query(downsampled_xyz)
    #n,d = downsampled_xyz.shape
    #downsampled_xyzrgb = np.zeros((n,d+3))
    #downsampled_xyzrgb[:,:d] = downsampled_xyz
    #downsampled_xyzrgb[:,d:] = good_rgb[i]

    n,d = good_xyz.shape
    xyzrgb = np.zeros((n,d+3))
    xyzrgb[:,:d] = good_xyz
    xyzrgb[:,d:] = good_rgb

    if returnOnlyXyz:
        return xyzrgb
    return (xyzrgb, rgb, T_w_k)

