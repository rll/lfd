#!/usr/bin/env python

import argparse
import h5py
from lfd.rapprentice.knot_classifier import isKnot
import sys
import os.path as osp

from string import lower

C_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
C_strs = ['1e-05', '0.0001', '0.001', '0.01', '1.0', '10.0', '100.0', '1000.0']
feature_types = ['base', 'mul', 'mul_s', 'mul_quad', 'landmark']
MODEL_TYPE='bellman'

def estimate_performance(results_file):
    if type(results_file) is str:
        results_file = h5py.File(results_file, 'r')

    num_knots = 0
    knot_inds = []
    not_inds = []
    ctr = 0
    n_checks = len(results_file) - 1
    for (i_task, task_info) in results_file.iteritems():
        sys.stdout.write("\rchecking task {} / {}        ".format(ctr, n_checks))
        sys.stdout.flush()
        ctr += 1
        if str(i_task) == 'args':
            continue
        # if int(i_task) > 3:
        #     break
        N_steps = len(task_info)
        final_cld = task_info[str(N_steps-1)]['next_state']['rope_nodes'][:]
        if isKnot(final_cld):
            knot_inds.append(int(i_task))
            num_knots += 1
        else:
            not_inds.append(int(i_task))

    print 
    
    return num_knots, knot_inds, not_inds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile")
    parser.add_argument("--baseline", type=str, default='../data/evals/7_3_0.1_baseline.h5')
    args = parser.parse_args()
    recompute_results = True
    if osp.exists(args.outfile):
        user_resp = raw_input("Overwrite results file {}[y/N]".format(args.outfile))
        recompute_results = lower(user_resp) == 'y'
    if recompute_results:
        outf = h5py.File(args.outfile, 'w')
        base_successes, a, b = estimate_performance(args.baseline)
        base_rate = base_successes / float(len(a) + len(b))
        outf['base_rate'] = base_rate
        for f in feature_types:
            for c in C_strs:
                results_fname = '../data/evals/jul_6_{}_0.1_c={}_{}.h5'.format(f, c, MODEL_TYPE)
                print "checking {}".format(results_fname)
    
                num_successes, knot_inds, not_inds = estimate_performance(results_fname)
                print "Success rate:", num_successes / float(len(knot_inds) + len(not_inds))
                key = str((f, float(c)))
                outf[key] = num_successes / float(len(knot_inds) + len(not_inds))
    else:
        outf = h5py.File(args.outfile, 'r')
    for c in C_strs:
        sys.stdout.write('\t\t{}'.format(c))
    print
    for f in feature_types:
        if f =='mul_quad' or f == 'landmark':
            sys.stdout.write('{}\t'.format(f))
        else:
            sys.stdout.write('{}\t\t'.format(f))
        for c in C_strs:
            sys.stdout.write('{:.2f}'.format(outf[str((f, float(c)))][()]))
            sys.stdout.write('\t\t')
        print
    sys.stdout.write('baseline\t')
    for c in C_strs:
        sys.stdout.write('{:.2f}'.format(outf['base_rate'][()]))
        sys.stdout.write('\t\t')
    print
    outf.close()
