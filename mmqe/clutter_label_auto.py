#!/usr/bin/env python

from __future__ import division

import copy
import argparse
from core import sim_util
from core.constants import ROPE_RADIUS, ROPE_RADIUS_THICK, MAX_ACTIONS_TO_TRY, NUM_PROCS

from core.transfer_simulate import BatchTransferSimulate
from core.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from core.simulation import ClutterSimulationRobotWorld

from core.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject
from core.environment import LfdEnvironment, GroundTruthRopeLfdEnvironment
from core.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory
from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer

from rapprentice import eval_util, util
from rapprentice import tps_registration

from rapprentice import plotting_openrave, task_execution, \
    resampling, ropesim, rope_initialization
from rapprentice.knot_classifier import remove_consecutive_crossings, remove_consecutive_cross_pairs, calculateCrossings, crossingsToString, isFig8Knot
import pdb
import time

import trajoptpy
import os
import os.path
import numpy as np
import h5py
from rapprentice.util import redprint, yellowprint
import atexit
import IPython as ipy
import random
from string import lower

OBJ_SPACING = 0.1
# Usage: python scripts/label.py --animation 2 label bigdata/misc/overhand_actions.h5 data/misc/Sep14_train2.h5 finger bij --gpu

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None

# @profile
def label_demos(args, transferer, lfd_env, sim):
    outfile = h5py.File(args.eval.outfile, 'a')

    rope_params = sim_util.RopeParams()
    #rope_params.radius = ROPE_RADIUS_THICK
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness

    #(init_rope_nodes, demo_id) = sample_init_state(
    #    sim, args.animation)
    sim.set_state(sample_init_state(sim))
    sim.settle(step_viewer=args.animation)
    resample = False


    labeled_data = []
    counter = 0
    any_success = False
    done = False


    for index in range(500):
        if args.animation:
            sim.viewer.Step()
        sim_state = sim.get_state()
        sim.set_state(sim_state)

        scene_state = lfd_env.observe_scene()

        # plot cloud of the test scene
        handles = []
        if args.plotting:
            handles.append(
                sim.env.plot3(
                    scene_state.cloud[
                        :,
                        :3],
                    2,
                    scene_state.color if scene_state.color is not None else (
                        0,
                        0,
                        1)))
            sim.viewer.Step()
        old_cleared_objs = sim.remove_cleared_objs()
        costs = transferer.registration_factory.batch_cost(scene_state)
        best_keys = sorted(costs, key=costs.get)
        #any_success = False
        #for seg_name in best_keys:
        for seg_index, seg_name in enumerate(best_keys):
            redprint(seg_name)
            traj = transferer.transfer(GlobalVars.demos[seg_name],
                                                scene_state,
                                                plotting=args.plotting)
            #resp = raw_input('continue?[Y/n]')
            #if resp in ('n', 'N'):
            #    continue
            feasible, misgrasp, grasped_objs = lfd_env.execute_augmented_trajectory(
                traj, step_viewer=args.animation, interactive=args.interactive, return_grasped_objs=True)
            reward = sim.compute_reward(old_cleared_objs, grasped_objs)
            print "reward:\t{}".format(reward)
            sim_util.reset_arms_to_side(sim)
            sim.settle(step_viewer=args.animation)

            if not feasible or misgrasp:
                print 'Feasible: ', feasible
                print 'Misgrasp: ', misgrasp
                #if misgrasp:
                #    sim.set_state(sim_state)
                #    continue
            #print "y accepts this action"
            #print "ys accepts this action and saves sequence"
            #print "s rejects this action and saves sequence"
            #print "n rejects this action"
            #print "r resamples rope state"
            #print "f to save this as a failure"
            #print "C-c safely quits"
            #user_input = lower(raw_input("What to do?"))
            if reward > 0:
                user_input = 'y'
            else:
                if seg_index == len(best_keys)-1:
                    done = True
                    redprint("WENT TRHOUGH ALL DEMOS - SAVING NOW")
                user_input = 'n'
                redprint("Failure, objs: " + str(counter))
            if user_input == 'y':
                counter +=1
                redprint("Success " + str(counter))
                if counter >= 5:
                  done = True
                any_success = True
                labeled_data.append((scene_state,seg_name, reward))
                break
            elif user_input == 'ys':
                labeled_data.append((scene_state,seg_name, reward))
                save_success(outfile, labeled_data)
                labeled_data = []
                sim.set_state(sample_init_state(sim))
                break
            elif user_input == 's':
                save_success(outfile, labeled_data)
                labeled_data = []
                sim.set_state(sample_init_state(sim))
                break
            elif user_input == 'n':
                sim.set_state(sim_state)
                continue
            elif user_input == 'r':
                break
            elif user_input == 'f':
                sim.set_state(sim_state)
                save_failure(outfile, lfd_env.observe_scene())
                continue
        if any_success and done:
            redprint("SAVING SUCCESSFUL AUTO LABELED SEGMENTS")
            save_success(outfile, labeled_data)
            any_success = False
            done = False
            counter = 0
            labeled_data = []
            sim.set_state(sample_init_state(sim))
        elif done:
            redprint("ALL FAILURE - NOT SAVING")
            any_success = False
            done = False
            counter = 0
            labeled_data = []
            sim.set_state(sample_init_state(sim))

def save_failure(outfile, failure_scene):
    key = get_next_failure_key(outfile)
    g = outfile.create_group(key)
    g['cloud_xyz'] = failure_scene.cloud
    outfile.flush()

def save_success(outfile, labeled_data):
    i_task = get_next_task_i(outfile)
    print 'Saving ' +str(len(labeled_data)) + 'step knot to results, task', i_task
    for i_step in range(len(labeled_data)):
        scene, action, reward = labeled_data[i_step]
        key = str((i_task, i_step))
        g = outfile.create_group(key)
        g['cloud_xyz'] = scene.cloud
        g['action'] = action
        g['reward'] = reward
    outfile.flush()

def get_next_failure_key(outfile):
    failure_inds = [int(k[1:]) for k in outfile.keys() if k.startswith('f')]
    if not failure_inds:
        return 'f0'
    return 'f' + str(max(failure_inds)+1)

def parse_key(key):
    # parsing hackery to get a tuple of ints from its str representation
    return [int(x) for x in key.strip('(').strip(')').strip(' ').split(',')]

def get_next_task_i(outfile):
    task_is = [parse_key(k)[0] for k in outfile.keys() if k.startswith('(')]
    if len(task_is) == 0:
        return 0
    return max(task_is) + 1

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

def rand_pose(x_max, y_max, z_offset=0):
    T = np.eye(4)
    T[0, 3] = x_max * np.random.rand() - x_max / 2
    T[1, 3] = y_max * np.random.rand() - y_max / 2
    T[2, 3] = z_offset

    # rand_d = np.random.rand(3)
    # rand_d = rand_d / np.linalg.norm(rand_d)
    # T[:3, :3] = openravepy.matrixFromAxisAngle(rand_d)[:3, :3]

    return T

def sample_init_state(sim, animation=False, viewer=None, human_check=True):
    success = False
    for i in range(1):

        objs = [sim.coil] + sim.small_boxes + sim.big_boxes
        container_pose = sim.container.get_pose()
        sim.initialize({sim.coil.name:container_pose.dot(rand_pose(1,1,10))}, step_viewer=0)
        #sim.initialize(dict([(obj.name, obj) for obj in objs]), step_viewer=0)
        base_x, base_y = sim.container.get_footprint()
        z_start = sim.container.get_height() * 2

        new_objs = []
        new_big = []
        new_small = []

        np.random.shuffle(objs)

        init_state = {}

        for i, obj in enumerate(objs):
            P = container_pose.dot(
                rand_pose(base_x,
                          base_y,
                          z_offset=i*OBJ_SPACING + z_start))
            init_state[obj.name] = P

        # TODO: remove other items in sim first? (like in replace rope)
        sim.initialize(init_state, step_viewer=10)
        sim.settle(step_viewer=animation)
        sim.settle(step_viewer=animation)
        cld = sim.observe_cloud()
        print 'simulation settled, observation size: {}'.format(cld.shape[0])
        if viewer is not None:
            from rapprentice import plotting_openrave
            handles = []
            handles.append(sim.env.plot3(cld, 3, (0, 1, 0, 1)))
            viewer.Idle()


        #if human_check:
            #resp = raw_input("Use this simulation?[Y/n]")
            #success = resp not in ('N', 'n')
        #else:
            #success = True

    return sim.get_state()


def parse_input_args():
    parser = util.ArgumentParser()

    parser.add_argument(
        "--animation",
        type=int,
        default=0,
        help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--plotting", type=int, default=1,
                        help="plots if animation != 0 and plotting != 0")
    parser.add_argument("--interactive", action="store_true",
                        help="step animation and optimization if specified")

    parser.add_argument(
        "--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument(
        "--window_prop_file", type=str, default='../.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # arguments for eval
    parser_eval = subparsers.add_parser('eval')

    parser_eval.add_argument('actionfile', type=str, nargs='?',
                             default='../bigdata/misc/overhand_actions.h5')
    parser_eval.add_argument('--taskfile', type=str, nargs='?',
                             default='../data/misc/Sep13_r0.1_n7_train.h5')
    parser_eval.add_argument('outfile', type=str, nargs='?')

    parser_eval.add_argument("--n_examples", type=int, default=1000)
    #parser_eval.add_argument("--min_rad", type=float, default="0.1",
    #                         help="min perturbation radius")
    #parser_eval.add_argument("--max_rad", type=float, default="0.1",
    #                         help="max perturbation radius")
    #parser_eval.add_argument(
    #    "--n_perturb_pts",
    #    type=int,
    #    default=7,
    #    help="number of points perturbed from demo start state")
    #parser_eval.add_argument("--dagger_states_file", type=str)
    #parser_eval.add_argument("--label_single_step", action="store_true")

    parser_eval.add_argument(
        "transferopt",
        type=str,
        nargs='?',
        choices=[
            'pose',
            'finger'],
        default='finger')
    parser_eval.add_argument(
        "reg_type", type=str, choices=['segment', 'rpm', 'bij'], default='bij')
    parser_eval.add_argument("--unified", type=int, default=0)

    parser_eval.add_argument("--downsample", type=int, default=1)
    parser_eval.add_argument("--downsample_size", type=float, default=0.012)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument(
        "--upsample_rad",
        type=int,
        default=1,
        help="upsample_rad > 1 incompatible with downsample != 0")
    parser_eval.add_argument("--ground_truth", type=int, default=0)

    parser_eval.add_argument(
        "--fake_data_segment", type=str, default='demo1-seg00')
    parser_eval.add_argument(
        "--fake_data_transform",
        type=float,
        nargs=6,
        metavar=(
            "tx",
            "ty",
            "tz",
            "rx",
            "ry",
            "rz"),
        default=[
            0,
            0,
            0,
            0,
            0,
            0],
        help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")

    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--check_feasible", type=int, default=0)

    parser_eval.add_argument("--alpha", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=100.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--use_collision_cost", type=int, default=1)

    parser_eval.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_param_radius", type=str, default=None)
    parser_eval.add_argument(
        "--rope_param_angStiffness", type=str, default=None)

    #parser_eval.add_argument("--parallel", action="store_true")
    parser_eval.add_argument("--gpu", action="store_true", default=False)

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
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(
        actions_root + '.cache' + actions_ext, 'a')

    GlobalVars.demos = {}
    for action, seg_info in GlobalVars.actions.iteritems():
        if args.eval.ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            scene_state = GroundTruthRopeSceneState(
                rope_nodes,
                ROPE_RADIUS,
                upsample=args.eval.upsample,
                upsample_rad=args.eval.upsample_rad,
                downsample_size=args.eval.downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
            scene_state = SceneState(
                full_cloud, downsample_size=args.eval.downsample_size)
        lr2arm_traj = {}
        lr2finger_traj = {}
        lr2ee_traj = {}
        lr2open_finger_traj = {}
        lr2close_finger_traj = {}
        for lr in 'lr':
            arm_name = {"l": "leftarm", "r": "rightarm"}[lr]
            lr2arm_traj[lr] = np.asarray(seg_info[arm_name])
            lr2finger_traj[lr] = sim_util.gripper_joint2gripper_l_finger_joint_values(
                np.asarray(
                    seg_info[
                        '%s_gripper_joint' %
                        lr]))[
                :,
                None]
            lr2ee_traj[lr] = np.asarray(
                seg_info["%s_gripper_tool_frame" % lr]['hmat'])
            lr2open_finger_traj[lr] = np.zeros(
                len(lr2finger_traj[lr]), dtype=bool)
            lr2close_finger_traj[lr] = np.zeros(
                len(lr2finger_traj[lr]), dtype=bool)
            opening_inds, closing_inds = sim_util.get_opening_closing_inds(
                lr2finger_traj[lr])
# opening_inds/closing_inds are indices before the opening/closing happens, so increment those indices (if they are not out of bound)
# opening_inds = np.clip(opening_inds+1, 0, len(lr2finger_traj[lr])-1) # TODO figure out if +1 is necessary
#             closing_inds = np.clip(closing_inds+1, 0, len(lr2finger_traj[lr])-1)
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
        aug_traj = AugmentedTrajectory(
            lr2arm_traj=lr2arm_traj,
            lr2finger_traj=lr2finger_traj,
            lr2ee_traj=lr2ee_traj,
            lr2open_finger_traj=lr2open_finger_traj,
            lr2close_finger_traj=lr2close_finger_traj)
        demo = Demonstration(action, scene_state, aug_traj)
        GlobalVars.demos[action] = demo


def setup_lfd_environment_sim(args):
    actions = h5py.File(args.eval.actionfile, 'r')

    sim = ClutterSimulationRobotWorld(2, 2)
    world = sim

    lfd_env = LfdEnvironment(sim, world, downsample_size=args.eval.downsample_size)

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

    return lfd_env, sim


def setup_registration_and_trajectory_transferer(args, sim):
    print 'Setting up registration'
    if args.eval.gpu:
        if args.eval.reg_type == 'rpm':
            reg_factory = GpuTpsRpmRegistrationFactory(
                GlobalVars.demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            print 'using GPU registration'
            reg_factory = GpuTpsRpmBijRegistrationFactory(
                GlobalVars.demos, args.eval.actionfile)
        else:
            raise RuntimeError(
                "Invalid reg_type option %s" % args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'bij':
            # print 'using CPU registration'
            reg_factory = TpsRpmBijRegistrationFactory(
                GlobalVars.demos, args.eval.actionfile)
        else:
            raise RuntimeError(
                "Invalid reg_type option %s" % args.eval.reg_type)

    print 'Setting up transferer'
    if args.eval.transferopt == 'pose' or args.eval.transferopt == 'finger':
        traj_transferer = PoseTrajectoryTransferer(
            sim,
            args.eval.beta_pos,
            args.eval.beta_rot,
            args.eval.gamma,
            args.eval.use_collision_cost)
        if args.eval.transferopt == 'finger':
            traj_transferer = FingerTrajectoryTransferer(
                sim,
                args.eval.beta_pos,
                args.eval.gamma,
                args.eval.use_collision_cost,
                init_trajectory_transferer=traj_transferer)
    else:
        raise RuntimeError("Invalid transferopt option %s" %
                           args.eval.transferopt)

    if args.eval.unified:
        reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(
            reg_factory, traj_transferer)
    else:
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(
            reg_factory, traj_transferer)
    return reg_and_traj_transferer


def main():
    args = parse_input_args()
    setup_log_file(args)

    set_global_vars(args)
    print 'Setting Global Vars'
    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args)
    reg_and_traj_transferer = setup_registration_and_trajectory_transferer(
        args, sim)

    if args.subparser_name == "eval":
        label_demos(args, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
