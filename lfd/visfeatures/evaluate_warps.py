#!/usr/bin/env python

import argparse, h5py, pickle, numpy as np
from operator import add
import IPython as ipy
from rapprentice import berkeley_pr2, clouds

corners_and_midpts = {}

def xy_to_xyz(xy, depth, T_w_k):
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

    (x,y) = xy
    return xyz_w[y][x]

def cloud_backto_xy(xyz, T_w_k):
    cx = 320. - .5
    cy = 240. - .5

    xyz_k = (xyz - T_w_k[:3,3][None,:]).dot(T_w_k[:3, :3])
    x = xyz_k[:, 0] / xyz_k[:, 2] * berkeley_pr2.f + cx
    y = xyz_k[:, 1] / xyz_k[:, 2] * berkeley_pr2.f + cy
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    return np.concatenate((x, y), axis=1)

def corner_midpt_cost(warped_source_c_m, target_c_m):
    (warped_source_c, warped_source_m) = warped_source_c_m
    (target_c, target_m) = [np.array(pts) for pts in target_c_m]
    corners_dist = 0
    for pt in warped_source_c:
        corners_dist += min(np.linalg.norm(target_c - pt, axis=1))

    midpts_dist = 0
    for pt in warped_source_m:
        midpts_dist += min(np.linalg.norm(target_m - pt, axis=1))

    return [corners_dist, midpts_dist]

def get_corners_midpts(corners_midpts_f_cloud, depth, T_w_k):
    corners = []
    midpts = []
    for k in corners_midpts_f_cloud:
        pt_type = corners_midpts_f_cloud[k]['type'][()]
        pt_xy = corners_midpts_f_cloud[k]['xy'][()]
        if pt_type == "corner":
            corners.append(xy_to_xyz(pt_xy, depth, T_w_k))
        elif pt_type == "midpoint":
            midpts.append(xy_to_xyz(pt_xy, depth, T_w_k))
        else:
            print "ERROR: point type is neither corner nor midpoint"

    return (corners, midpts)

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_cloud_type(source_id):
    if is_integer(source_id[-2:]):
        return source_id[:-2]
    else:
        return source_id[:-1]

def calc_corner_midpt_costs(warp_fns_and_costs, corners_midpts_f, rgb_f):
    global corners_and_midpts

    sum_costs = {}
    i = 0
    target_ids_seen = set()
    indiv_costs = {}
    for (use_vis, source_id, target_id, fn, cost) in warp_fns_and_costs:
        if target_id not in target_ids_seen:
            print target_id
            target_ids_seen.add(target_id)
        cloud_type = get_cloud_type(source_id)
        if get_cloud_type(target_id) != cloud_type:
            continue

        i += 1
        if cloud_type not in sum_costs:
            sum_costs[cloud_type] = {}
            indiv_costs[cloud_type] = {}
        if use_vis not in sum_costs[cloud_type]:
            sum_costs[cloud_type][use_vis] = [0,0,0,0]
            indiv_costs[cloud_type][use_vis] = []

        if source_id not in corners_midpts_f or target_id not in corners_midpts_f:
            sum_costs[cloud_type][use_vis] = map(add, sum_costs[cloud_type][use_vis], [0,0,cost,1])
            continue

        if source_id not in corners_and_midpts:
            corners_and_midpts[source_id] = get_corners_midpts(corners_midpts_f[source_id],
                    rgb_f[source_id]['depth'][()], rgb_f[source_id]['T_w_k'][()])
        source_c_m = corners_and_midpts[source_id]
        warped_source_c = [fn.transform_points(np.array([pt])) for pt in source_c_m[0]]
        warped_source_m = [fn.transform_points(np.array([pt])) for pt in source_c_m[1]]
        warped_source_c_m = (warped_source_c, warped_source_m)

        if target_id not in corners_and_midpts:
            corners_and_midpts[target_id] = get_corners_midpts(corners_midpts_f[target_id],
                    rgb_f[target_id]['depth'][()], rgb_f[target_id]['T_w_k'][()])
        target_c_m = corners_and_midpts[target_id]

        costs = corner_midpt_cost(warped_source_c_m, target_c_m)
        sum_costs[cloud_type][use_vis] = map(add, sum_costs[cloud_type][use_vis], costs + [cost, 1])
        indiv_costs[cloud_type][use_vis].append([use_vis, source_id, target_id] + costs + [cost, 1])

    top_20_sum = []
    for cloud_type in indiv_costs.keys():
        sorted_novis = sorted([(vals[3], i) for (i,vals) in enumerate(indiv_costs[cloud_type][0])], key=lambda x: x[0])
        sorted_vis = sorted([(vals[3], i) for (i,vals) in enumerate(indiv_costs[cloud_type][1])], key=lambda x: x[0])

        no_vis_sum = (sum([indiv_costs[cloud_type][0][i][3] for (_,i) in sorted_novis[:20]]), sum([indiv_costs[cloud_type][0][i][4] for (_,i) in sorted_novis[:20]]))
        vis_sum = (sum([indiv_costs[cloud_type][1][i][3] for (_,i) in sorted_novis[:20]]), sum([indiv_costs[cloud_type][1][i][4] for (_,i) in sorted_novis[:20]]))
        top_20_sum.append([cloud_type, no_vis_sum, vis_sum])

    #ipy.embed()

    print "Total num considered:", i
    print "Top 20 sum", top_20_sum
    return sum_costs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_warps", type=str)
    parser.add_argument("labeled_corners_and_midpts", type=str)
    parser.add_argument("rgb_depth_file", type=str)
    args = parser.parse_args()

    corners_midpts_f = h5py.File(args.labeled_corners_and_midpts, 'r')
    rgb_f = h5py.File(args.rgb_depth_file, 'r')
    with open(args.saved_warps, 'rb') as f:
        warp_fns_and_costs = pickle.load(f)

    costs = calc_corner_midpt_costs(warp_fns_and_costs, corners_midpts_f, rgb_f)
    print "Final costs:"
    print costs

    corners_midpts_f.close()
    rgb_f.close()

if __name__ == "__main__":
    main()
