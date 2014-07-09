"""
Utility functions for manipulation rope data and features
"""

import numpy as np
from constants import GRIPPER_OPEN_CLOSE_THRESH

def get_closing_pts(seg_info, as_dict = False):
    closing_inds = get_closing_inds(seg_info)
    closing_pts = []
    if as_dict: closing_pts = {}
    for lr in closing_inds:
        if closing_inds[lr] != -1:
            hmat = seg_info["%s_gripper_tool_frame"%lr]['hmat'][closing_inds[lr]]
            if not as_dict:
                closing_pts.append(extract_point(hmat))
            else:
                closing_pts[lr] = extract_point(hmat)
    return closing_pts

def get_closing_inds(seg_info):
    """
    returns a dictionary mapping 'l', 'r' to the index in the corresponding trajectory
    where the gripper first closes
    """
    result = {}
    for lr in 'lr':
        grip = np.asarray(seg_info[lr + '_gripper_joint'])
        closings = np.flatnonzero((grip[1:] < GRIPPER_OPEN_CLOSE_THRESH) \
                                      & (grip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))
        if closings:
            result[lr] = closings[0]
        else:
            result[lr] = -1
    return result
