#!/usr/bin/env python
#
# Run this script with:
#     ./filter_labeled_examples.py <labeled_ex_file> <output_ex_file>
# Include flag --remove_short_traj to remove trajectories with length < 4.
# Include flag --remove_deadend_traj to remove trajectories that end in a
# deadend (as opposed to a knot).
# Include flag --remove_long_traj to remove trajectories with length > 5.
#
# Outputs a new labeled examples file, excluding trajectories depending on which
# flags are set (see above explanation). Renumbers the trajectories so that
# their ids are in consecutive numerical order, starting from 0.
# Assumes ids of the labelled examples are in consecutive numerical order,
# starting from 0.

import argparse, h5py

def filter_labeled_examples(examples, output, remove_short, remove_long,
                            remove_deadend):
    num_examples = len(examples.keys())
    if num_examples == 0:
        return

    output_id = 0  # To keep track of renumbering
    prev_start = 0
    for i in range(num_examples):
        k = str(i)
        pred = int(examples[k]['pred'][()])
        if pred == i and i != 0:
            to_remove = False
            if remove_short and i - prev_start < 4:
                # trajectory has less than 4 steps, inc. endstate
                to_remove = True
            if remove_long and i - prev_start > 5:
                to_remove = True
            if remove_deadend and 'deadend' in examples[str(i-1)].keys() and \
                    examples[str(i-1)]['deadend'][()] == 1:
                # trajectory ends in a deadend
                to_remove = True
            if to_remove:
                print "Removing trajectory starting at id ", prev_start, ", length: ", i - prev_start
                for i_rm in range(i - prev_start):
                    output_id -= 1
                    print "Deleting output id ", output_id
                    del output[str(output_id)]
                print "Adding again at output id ", output_id
            prev_start = i

        new_group = output.create_group(str(output_id))
        for group_key in examples[k].keys():
            # Update the value of 'pred' correctly (with the renumbering)
            if group_key == 'pred':
                assert pred == i or pred == i-1, "Invalid predecessor value for %i"%i
                if pred == i:
                    new_group[group_key] = str(output_id)
                else:
                    new_group[group_key] = str(output_id - 1)
            else:
                new_group[group_key] = examples[k][group_key][()]
        output_id += 1

    to_remove = False
    if remove_short and num_examples - prev_start < 4:
        to_remove = True
    if remove_long and num_examples - prev_start > 5:
        to_remove = True
    if remove_deadend and 'deadend' in examples[str(num_examples-1)].keys() and \
            examples[str(num_examples-1)]['deadend'][()] == 1:
        to_remove = True
    if to_remove:      
        print "Removing trajectory starting at id ", prev_start, ", length: ", num_examples - prev_start
        for i_rm in range(num_examples - prev_start):
            output_id -= 1
            print "Deleting output id ", output_id
            del output[str(output_id)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file')
    parser.add_argument('output_examples_file')
    parser.add_argument('--remove_short_traj', action='store_true')
    parser.add_argument('--remove_long_traj', action='store_true')
    parser.add_argument('--remove_deadend_traj', action='store_true')
    args = parser.parse_args()

    examples = h5py.File(args.examples_file, 'r')
    output = h5py.File(args.output_examples_file, 'w')
    filter_labeled_examples(examples, output, args.remove_short_traj,
                            args.remove_long_traj, args.remove_deadend_traj)
    examples.close()
    output.close()
