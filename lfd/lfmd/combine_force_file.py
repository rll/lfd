import argparse
import h5py
import pickle
import numpy as np
import IPython as ipy

parser = argparse.ArgumentParser()
parser.add_argument("actionfile", type=str, help="h5 file that has the action information to which the forces needs to be added")
parser.add_argument("forcefile", type=str, help="pickle file that has the force information")
args = parser.parse_args()

actions = h5py.File(args.actionfile)
forcefile = open(args.forcefile, "rb")
forces = pickle.load(forcefile)

for action, seg_info in actions.iteritems():
    for lr in 'lr':
        seg_info["%s_gripper_force"%lr] = np.asarray(forces[action][lr])

actions.close()
forcefile.close()
