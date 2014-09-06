"""
scripts for build constraint files and models and for optimizing models
"""

from features import BatchRCFeats, MulFeats, QuadMulFeats, SimpleMulFeats, LandmarkFeats
from constraints import ConstraintGenerator, BatchCPMargin
from max_margin import MaxMarginModel, BellmanMaxMarginModel

import os.path as osp
import h5py, sys
from string import lower

import IPython as ipy

class State(object):
    """
    placeholder state
    """
    def __init__(self, id, cloud):
        self.id = id
        self.cloud = cloud

def get_feat_cls(args):
    if args.feature_type == 'base':
        return BatchRCFeats
    elif args.feature_type == 'mul':
        return MulFeats
    elif args.feature_type == 'mul_s':
        return SimpleMulFeats
    elif args.feature_type == 'mul_quad':
        return QuadMulFeats
    elif args.feature_type == 'landmark':
        return LandmarkFeats

def get_model_cls(args):
    if args.model_type == 'bellman':
        return BellmanMaxMarginModel
    else:
        return MaxMarginModel

def get_constr_generator(args):
    feats = get_feat_cls(args)(args.actionfile)
    try:
        feats.set_landmark_file(args.landmarkfile)
    except AttributeError:
        pass
    marg  = BatchCPMargin(feats)
    constr_gen = ConstraintGenerator(feats, marg, args.actionfile)
    return constr_gen

def check_exists(fname):
    if osp.exists(fname):
        resp = raw_input("Really Overwrite File {}?[y/N]".format(fname))
        if lower(resp) != 'y':
            return False
    return True

def get_actions(args):
    f = h5py.File(args.actionfile, 'r')
    actions = f.keys()
    f.close()
    return actions


def build_constraints(args):
    print "Loading Constraints from {} to {}".format(args.demofile, args.constrfile)
    start = time.time()
    constr_generator = get_constr_generator(args)
    exp_demofile = h5py.File(args.demofile, 'r')
    if not check_exists(args.constrfile):
        return
    constrfile = h5py.File(args.constrfile, 'w')
    n_constraints = len(exp_demofile)        
    for i, demo_k in enumerate(exp_demofile):
        ## we expect states to be an identifier and a 
        ## point cloud, we won't use the identifier here
        sys.stdout.write('\rcomputing constraints {}/{}       '.format(i, n_constraints))
        sys.stdout.flush()
        demo_info = exp_demofile[demo_k]
        exp_a = demo_info['action'][()]
        if exp_a.startswith('endstate'): # this is a knot
            continue
        state = State(i, demo_info['cloud_xyz'][:])
        exp_phi, phi, margins = constr_generator.compute_constrs(state, exp_a)
        constr_generator.store_constrs(exp_phi, phi, margins, exp_a, constrfile, constr_k=str(demo_k))
        constrfile.flush()
    print ""
    print "Constraint Generation Complete\nTime Taken:\t{}".format(time.time() - start)
    constrfile.close()
    exp_demofile.close()

def build_model(args):
    if not check_exists(args.modelfile):
        return
    print 'Building model into {}.'.format(args.modelfile)
    actions = get_actions(args)
    start = time.time()
    N = len(actions)
    feat_cls = get_feat_cls(args)
    mm_model = get_model_cls(args)(actions, feat_cls.get_size(N))
    mm_model.load_constraints_from_file(args.constrfile, max_constrs=args.max_constraints)
    mm_model.save_model(args.modelfile)
    print "Model Created and Saved\nTime Taken:\t{}".format(time.time() - start)

def optimize_model(args):
    import gurobipy as grb
    print 'Found model: {}'.format(args.modelfile)
    actions = get_actions(args)
    feat_cls = get_feat_cls(args)
    mm_model = get_model_cls(args).read(args.modelfile, actions, feat_cls.get_size(len(actions)))
    try:
        mm_model.scale_objective(args.C, args.D)
    except TypeError:
        mm_model.scale_objective(args.C)

    # mm_model.model.setParam('threads', 1)  # Use single thread instead of maximum
    # # barrier method (#2) is default for QP, but uses more memory and could lead to error
    # # mm_model.model.setParam('method', 1)  # Use dual simplex method to solve model
    # mm_model.model.setParam('method', 0)  # Use primal simplex method to solve model
    try:
        mm_model.model.setParam('method', 2)  # try solving model with barrier
        mm_model.optimize_model()
        mm_model.model.setParam('method', 2)  # Use dual simplex method to solve model
        assert mm_model.model.status == 2
    except grb.GurobiError, AssertionError:
        print "model failure"
        mm_model.model.setParam('threads', 1)  # Use single thread instead of maximum
        mm_model.model.setParam('method', 0)  # Use primal simplex method to solve model
        mm_model.optimize_model()
    assert mm_model.model.status == 2
    mm_model.save_weights_to_file(args.weightfile)

def do_all(args):
    model_dir = '../data/models'
    weights_dir = '../data/weights'
    _, demofname = osp.split(args.demofile)
    labels = osp.splitext(demofname)[0]
    args.constrfile = '{}/{}_{}.h5'.format(model_dir, labels, args.feature_type)
    args.modelfile=osp.splitext(args.constrfile)[0] + '_{}.mps'.format(args.model_type)
    build_constraints(args)
    build_model(args)
    c_vals = args.C
    for c in c_vals:
        args.weightfile='{}/{}_{}_c={}_{}.h5'.format(weights_dir, labels, args.feature_type, c, args.model_type)        
        args.C = c
        optimize_model(args)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_type", type=str, choices=['base', 'mul', 'mul_quad', 'mul_s', 'landmark'])
    parser.add_argument("model_type", type=str, choices=['max-margin', 'bellman'])
    parser.add_argument("landmarkfile", type=str, nargs='?', default='../data/misc/landmarks.h5')
    subparsers = parser.add_subparsers()
    # build-constraints subparser
    parser_build_constraints = subparsers.add_parser('build-constraints')
    parser_build_constraints.add_argument('demofile',   nargs='?', default='../data/labels/all_labels_fixed.h5')
    parser_build_constraints.add_argument('constrfile', nargs='?', default='../data/models/all_labels_fixed.h5')
    parser_build_constraints.add_argument('actionfile', nargs='?', default='../data/misc/actions.h5')
    parser_build_constraints.set_defaults(func=build_constraints)

    # build-model subparser
    parser_build_model = subparsers.add_parser('build-model')
    parser_build_model.add_argument('constrfile', nargs='?', default='../data/models/all_labels_fixed.h5')
    parser_build_model.add_argument('modelfile',  nargs='?', default='../data/models/all_labels_fixed.mps')
    parser_build_model.add_argument('actionfile', nargs='?', default='../data/misc/actions.h5')
    parser_build_model.add_argument('--max_constraints', type=int, default=1000)
    parser_build_model.set_defaults(func=build_model)

    # optimize-model subparser
    parser_optimize = subparsers.add_parser('optimize-model')
    parser_optimize.add_argument('--C', '-c', type=float, default=1)
    parser_optimize.add_argument('--D', '-d', type=float, default=1)
    parser_optimize.add_argument('--F', '-f', type=float, default=1)
    parser_optimize.add_argument('modelfile',  nargs='?', default='../data/models/all_labels_fixed.mps')
    parser_optimize.add_argument('weightfile', nargs='?', default='../data/weights/all_labels_fixed.h5')
    parser_optimize.add_argument('actionfile', nargs='?', default='../data/misc/actions.h5')
    parser_optimize.set_defaults(func=optimize_model)

    parser_all = subparsers.add_parser('full')
    parser_all.add_argument('demofile',   nargs='?', default='../data/labels/labels_Jul_3_0.1.h5')
    parser_all.add_argument('actionfile', nargs='?', default='../data/misc/actions.h5')
    parser_all.add_argument('--max_constraints', type=int, default=1000)
    parser_all.add_argument('--C', '-c', type=float, nargs='*', default=[1])
    parser_all.add_argument('--D', '-d', type=float, default=1)
    parser_all.add_argument('--F', '-f', type=float, default=1)
    parser_all.set_defaults(func=do_all)
    return parser.parse_args()
    
if __name__=='__main__':
    import time
    args = parse_arguments()
    args.func(args)
