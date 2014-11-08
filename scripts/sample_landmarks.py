#!/usr/bin/env python
#
# Run this script with:
#     ./filter_labeled_examples.py <labeled_ex_file> <new_landmark_file>
#
# Make sure to run precompute without fill_traj on the example file first.

import argparse, h5py
import random

def gen_landmarks(examples, landmarks):
    failure_keys = [k for k in examples.keys() if k.startswith('f')]
    landmark_keys = [k for k in examples.keys() if k.startswith('(')]

    print len(failure_keys)

    sampled_keys = random.sample(failure_keys, 20) + random.sample(landmark_keys, 50)

    for key in sampled_keys:
        landmarks.copy(examples[key], key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file')
    parser.add_argument('landmark_file')
    args = parser.parse_args()

    examples = h5py.File(args.examples_file, 'r')
    landmarks = h5py.File(args.landmark_file, 'a')
    gen_landmarks(examples, landmarks)
    examples.close()
    landmarks.close()
