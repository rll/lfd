#!/usr/bin/env python

import argparse, h5py, pickle, os, numpy as np
import IPython as ipy
import vis_utils

P_R_FOLDER = 'saved_precision_and_recalls_rope'
LABELDIST_MULT = 1
SOURCE_LABELS_FILE = '/home/shhuang/research/data_towel/iros_rope_source_2conv3fc4_labels.h5'
TARGET_LABELS_FILE = '/home/shhuang/research/data_towel/iros_rope_target_2conv3fc4_labels.h5'

def get_avg_precision_recall(k_precision_recall):
    avg_precisions = None
    avg_recalls = None

    for c in k_precision_recall:
        if avg_precisions == None:
            avg_precisions = k_precision_recall[c][0]
            avg_recalls = k_precision_recall[c][1]
        else:
            avg_precisions = [sum(x) for x in zip(avg_precisions, k_precision_recall[c][0])]
            avg_recalls = [sum(x) for x in zip(avg_recalls, k_precision_recall[c][1])]

    avg_precisions = [x / float(len(k_precision_recall)) for x in avg_precisions]
    avg_recalls = [x / float(len(k_precision_recall)) for x in avg_recalls]
    return (avg_precisions, avg_recalls)

def print_k_vs_precision_table(avg_precisions):
    print "K\tPrecision"
    for i in xrange(len(avg_precisions)):
        print str(i+1) + "\t" + str(avg_precisions[i])

def print_recall_vs_precision_table(avg_precisions, avg_recalls):
    print "Recall\tPrecision"
    for i in xrange(len(avg_precisions)):
        print str(avg_recalls[i]) + "\t" + str(avg_precisions[i])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_warp_fn_and_costs', type=str)
    parser.add_argument('ground_truth_warps', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--use_vis', type=int, default=0)
    args = parser.parse_args()

    with open(args.saved_warp_fn_and_costs, 'rb') as f:
        warp_fns_and_costs = pickle.load(f)

    ground_truth = h5py.File(args.ground_truth_warps, 'r')

    source_labels = None
    target_labels = None
    if args.use_vis != 0:
        source_labels = h5py.File(SOURCE_LABELS_FILE, 'r')
        target_labels = h5py.File(TARGET_LABELS_FILE, 'r')

    costs = {}
    for (use_vis, source_id, target_id, fn, cost) in warp_fns_and_costs:
        if use_vis != args.use_vis:  # Only pay attention to type of TPS-RPM specified
            continue
        if target_id not in ground_truth:
            continue

        if target_id not in costs:
            costs[target_id] = []

        if source_labels != None:
            labels_dist = vis_utils.get_labels_distance(source_labels[source_id]['predicts_ds'][()], target_labels[target_id]['predicts_ds'][()])
            cost = cost + LABELDIST_MULT * labels_dist

        costs[target_id].append((cost, source_id))

    if args.use_vis != 0:
        source_labels.close()
        target_labels.close()

    k_precision_recall = {}
    for c in costs:
        ranked_source_ids = [x[1] for x in sorted(costs[c])]
        precisions = [1]
        recalls = [0]
        num_valid_in_top_k = 0

        for i in xrange(len(ranked_source_ids)):
            if ranked_source_ids[i] in ground_truth[c].keys():
                num_valid_in_top_k += 1
            precisions.append(num_valid_in_top_k / float(i+1))
            recalls.append(num_valid_in_top_k / float(len(ground_truth[c].keys())))

        # Use interpolated precision
        for i in xrange(len(precisions)):
            precisions[i] = max(precisions[i:])

        k_precision_recall[c] = (precisions, recalls)

    (average_precisions, average_recalls) = get_avg_precision_recall(k_precision_recall)
    #print_k_vs_precision_table(average_precisions)
    #print_recall_vs_precision_table(average_precisions, average_recalls)

    with open(os.path.join(P_R_FOLDER, args.output_file), 'wb') as f:
        pickle.dump((average_precisions, average_recalls), f)

if __name__ == "__main__":
    main()
