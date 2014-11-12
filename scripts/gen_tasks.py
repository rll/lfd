#!/usr/bin/env python

from __future__ import division

import time
import os.path
import h5py
import atexit
import random
import sys

import trajoptpy
import numpy as np

from lfd.environment import sim_util
from lfd.environment.simulation import DynamicSimulation
from lfd.environment.simulation_object import BoxSimulationObject, RopeSimulationObject
from lfd.rapprentice import util
from lfd.rapprentice import task_execution, rope_initialization
from lfd.rapprentice.util import redprint

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None


def replace_rope(sim, new_rope, animation):
    rope_sim_obj = None
    for sim_obj in sim.sim_objs:
        if isinstance(sim_obj, RopeSimulationObject):
            rope_sim_obj = sim_obj
            break
    if rope_sim_obj:
        sim.remove_objects([rope_sim_obj])
    rope = RopeSimulationObject("rope", new_rope, sim_util.RopeParams())
    sim.add_objects([rope])
    sim.settle(step_viewer=animation)


def load_random_start_segment(demofile):
    start_keys = [seg for seg in GlobalVars.actions.keys() if 'seg00' in seg]
#    start_keys = [seg for seg in GlobalVars.actions.keys() if seg.startswith('demo') and 'seg00' in seg]
    seg_name = random.choice(start_keys)
    return (GlobalVars.actions[seg_name]['cloud_xyz'], seg_name)


def sample_rope_state(demofile, sim, animation, human_check=False,
                      perturb_points=7, min_rad=0.1, max_rad=0.1):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        new_xyz, demo_id = load_random_start_segment(demofile)
        perturb_radius = random.uniform(min_rad, max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                        perturb_peak_dist=perturb_radius,
                                                                        num_perturb_points=perturb_points)
        replace_rope(sim, rope_nodes, animation)
        sim.settle()
        if animation:
            sim.viewer.Step()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True
    return (rope_nodes, demo_id)


def gen_task_file(args, sim, rotation_angle=0):
    """
    draw n_examples states from the initial state distribution defined by
    sample_rope_state using random intial states from actionfile

    writes results to task file name

    TODO: Add rotation angle to available perturbation
    """
    taskfile = h5py.File(args.gen_tasks.taskfile, 'w')
    actionfile = h5py.File(args.gen_tasks.actionfile, 'r')
    try:
        for i in range(args.gen_tasks.n_examples):
            redprint('Creating State {}/{}'.format(i, args.gen_tasks.n_examples))
            (rope_nodes, demo_id) = sample_rope_state(actionfile, sim,
                                                      args.animation,
                                                      human_check=args.interactive,
                                                      perturb_points=args.gen_tasks.n_perturb_pts,
                                                      min_rad=args.gen_tasks.min_rad,
                                                      max_rad=args.gen_tasks.max_rad)
            taskfile.create_group(str(i))
            taskfile[str(i)]['rope_nodes'] = rope_nodes
            taskfile[str(i)]['demo_id'] = str(demo_id)

        taskfile.create_group('args')
        taskfile['args']['num_examples'] = args.gen_tasks.n_examples
        taskfile['args']['actionfname'] = args.gen_tasks.actionfile
        taskfile['args']['perturb_bounds'] = (args.gen_tasks.min_rad,
                                              args.gen_tasks.max_rad)
        taskfile['args']['num_perturb_pts'] = args.gen_tasks.n_perturb_pts
        taskfile['args']['rotation'] = float(rotation_angle)
        print ''
    except:
        print 'encountered exception', sys.exc_info()
        raise
    finally:
        taskfile.close()
        actionfile.close()
    assert check_task_file(args.gen_tasks.taskfile, args.gen_tasks.n_examples)


def check_task_file(fname, n_examples):
    """
    probably unecessary, but checks that a task file is properly labelled sequentially
    """
    f = h5py.File(fname, 'r')
    success = True
    for i in range(n_examples):
        if str(i) not in f:
            print 'task file {} is missing key {}'.format(fname, i)
            success = False
    f.close()
    return success


def parse_input_args():
    parser = util.ArgumentParser()

    parser.add_argument("--animation", type=int, default=0,
                        help="animates if non-zero. viewer is stepped according to this number")
    parser.add_argument("--interactive", action="store_true",
                        help="Ask for human confirmation after each new rope state")

    parser.add_argument("--camera_matrix_file", type=str,
                        default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str,
                        default='../.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # arguments for eval
    parser_eval = subparsers.add_parser('gen_tasks')

    parser_eval.add_argument('actionfile', type=str, nargs='?',
                             default='../bigdata/misc/overhand_actions.h5')
    parser_eval.add_argument('taskfile', type=str, nargs='?')


    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6,
                             metavar=("tx","ty","tz","rx","ry","rz"),
                             default=[0,0,0,0,0,0],
                             help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")

    parser_eval.add_argument("--n_examples", type=int, default=100)
    parser_eval.add_argument("--min_rad", type=float, default="0.1",
                             help="min perturbation radius")
    parser_eval.add_argument("--max_rad", type=float, default="0.1",
                             help="max perturbation radius")
    parser_eval.add_argument("--n_perturb_pts", type=int, default=5,
                             help="number of points perturbed from demo start state")


    args = parser.parse_args()
    if not args.animation:
        args.plotting = 0
    return args

def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args):
    if args.random_seed is not None: np.random.seed(args.random_seed)
    GlobalVars.actions = h5py.File(args.gen_tasks.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.gen_tasks.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')

def setup_lfd_environment_sim(args):
    actions = h5py.File(args.gen_tasks.actionfile, 'r')

    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, args.gen_tasks.fake_data_segment, args.gen_tasks.fake_data_transform)
    table_height = init_rope_xyz[:,2].mean() - .02
    sim_objs = []
    sim_objs.append(BoxSimulationObject("table",
                                        [1, 0, table_height + (-.1 + .01)],
                                        [.85, .85, .1], dynamic=False))

    sim = DynamicSimulation()
    world = sim
    sim.add_objects(sim_objs)

    if args.animation:
        viewer = trajoptpy.GetViewer(sim.env)
        if os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                viewer.SetWindowProp(*window_prop)
                viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = viewer.GetWindowProp()
                camera_matrix = viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        viewer.Step()

    return sim

def main():
    args = parse_input_args()

    setup_log_file(args)

    set_global_vars(args)
    sim = setup_lfd_environment_sim(args)

    if args.subparser_name == "gen_tasks":
        start = time.time()
        gen_task_file(args, sim)
        print "time is:\t{}".format(time.time() - start)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
