#!/usr/bin/env python

import argparse, h5py, pickle, numpy as np
from operator import add
import IPython as ipy
from rapprentice import berkeley_pr2, clouds

FIRST_STEP = set(['demo00001_seg00', 'demo00002_seg00', 'demo00003_seg00', 'demo00003_seg01', 'demo00004_seg00', 'demo00004_seg01', 'demo00005_seg00', 'demo00006_seg00', 'demo00007_seg00', 'demo00007_seg01', 'demo00008_seg00', 'demo00009_seg00', 'demo00010_seg00'])
SECOND_STEP = set(['demo00001_seg01', 'demo00002_seg01', 'demo00005_seg01', 'demo00006_seg01', 'demo00008_seg01', 'demo00009_seg01', 'demo00010_seg01', 'demo00003_seg02', 'demo00004_seg02', 'demo00007_seg02'])
THIRD_STEP = set(['demo00001_seg02','demo00002_seg02','demo00003_seg03','demo00004_seg03','demo00005_seg02','demo00006_seg02','demo00007_seg03','demo00008_seg02','demo00009_seg02','demo00010_seg02'])

def calc_corr_cost(warp_fns_and_costs, ground_truth):
    pairs_with_ground_truth = set()
    for target_key in ground_truth:
        for source_key in ground_truth[target_key].keys():
            pairs_with_ground_truth.add((source_key, target_key))

    sum_costs = {}
    sum_costs['first'] = {}
    sum_costs['second'] = {}
    sum_costs['third'] = {}
    for k in sum_costs.keys():
        sum_costs[k][0] = [0,0,0,0]
        sum_costs[k][1] = [0,0,0,0]
    checked = 0
    for (use_vis, source_key, target_key, fn, cost) in warp_fns_and_costs:
        if (source_key, target_key) not in pairs_with_ground_truth:
            continue

        if target_key in FIRST_STEP:
            step = 'first'
        elif target_key in SECOND_STEP:
            step = 'second'
        elif target_key in THIRD_STEP:
            step = 'third'
        else:
            print "ERROR, key ", target_key, "not found"
            
        checked += 1

        gt_corr = ground_truth[target_key][source_key]['corr'][()]
        gt_cost = ground_truth[target_key][source_key]['cost'][()]
        target_pts = ground_truth[target_key][source_key]['target_pts'][()]
        source_pts = ground_truth[target_key][source_key]['source_pts'][()]
        lowest_warp_diff = ground_truth[target_key][source_key]['lowest_warp_diff'][()]

        warped_source = fn.transform_points(source_pts)
        n,d = warped_source.shape
        warp_diff = sum([np.linalg.norm(np.subtract(warped_source[i,:], target_pts[i,:])) for i in range(n)])

        sum_costs[step][use_vis] = map(add, sum_costs[step][use_vis], [warp_diff, warp_diff - lowest_warp_diff, cost, 1])
        
    print "Successfully checked", checked, "out of", len(warp_fns_and_costs)
    return sum_costs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_warps", type=str)
    parser.add_argument("ground_truth_warps", type=str)
    args = parser.parse_args()

    with open(args.saved_warps, 'rb') as f:
        warp_fns_and_costs = pickle.load(f)

    ground_truth = h5py.File(args.ground_truth_warps, 'r')
    costs = calc_corr_cost(warp_fns_and_costs, ground_truth)
    print "Final costs:"
    print costs

    ground_truth.close()

if __name__ == "__main__":
    main()
