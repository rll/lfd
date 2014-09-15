#!/usr/bin/env python

from __future__ import division

import copy
import argparse
from core import sim_util
from core.constants import ROPE_RADIUS, MAX_ACTIONS_TO_TRY

from core.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from core.simulation import DynamicSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject
from core.environment import LfdEnvironment, GroundTruthRopeLfdEnvironment
from core.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory
from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer

from rapprentice import eval_util, util
from rapprentice import tps_registration

from rapprentice import plotting_openrave, task_execution, \
    resampling, ropesim, rope_initialization
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


# Usage: python scripts/label.py --animation 2 label bigdata/misc/overhand_actions.h5 data/misc/Sep14_train2.h5 finger bij --gpu

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None

STEPS = ['init', '0', '1', '2', '3', '4']


def label_demos(args, transferer, lfd_env, sim):

    rope_params = sim_util.RopeParams()
    if args.label.rope_param_radius is not None:
        rope_params.radius = args.label.rope_param_radius
    if args.label.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.label.rope_param_angStiffness

    if args.label.dagger_states_file:
        dagger_states = h5py.File(args.label.dagger_states_file, 'r')
        task_indices = sorted(dagger_states.keys())
        curr_task_index = 0
        curr_step_index = 0
        load_dagger_state(dagger_states["0"], sim, args.animation)
        use_dagger = True
    else:
        # TODO pass in rope params to sample_rope_state
        (init_rope_nodes, demo_id) = sample_rope_state(
            args.label, sim, args.animation)
        use_dagger = False
    resample = False

    outfile = h5py.File(args.label.outfile, 'a')
    pred = str(len(outfile))
    #task_items = eval_util.get_indexed_items(taskfile, i_start=args.i_start, i_end=args.i_end)

    try:
        while True:
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

            (pred, resample) = manual_select_demo(args, transferer, sim,
                                                  lfd_env, outfile, pred)

            if resample and use_dagger:
                eof = False
                use_state = False
                while not use_state:
                    curr_step_index += 1
                    if curr_step_index == len(STEPS) or STEPS[curr_step_index] not in dagger_states[
                            task_indices[curr_task_index]]:
                        curr_task_index += 1
                        curr_step_index = 0
                        if curr_task_index == len(task_indices):
                            dagger_states.close()
                            eof = True
                            break
                    print "LOADING NEW DAGGER-SAMPLED STATE"
                    use_state = load_dagger_state(
                        dagger_states[
                            task_indices[curr_task_index]][
                            STEPS[curr_step_index]])
                if eof:
                    break
            elif resample:
                sample_rope_state(args.label, sim, args.animation)
    except KeyboardInterrupt:
        h5_no_endstate_len(outfile)
        safe = check_outfile(outfile, use_dagger)
        if not safe:
            print args.label.outfile + " is not properly formatted, check it manually!!!!!"


def get_input(start_scene, action_name, next_scene, outfile, pred):
    print "d accepts as knot and resamples rope"
    print "x accepts as deadend and resamples rope"
    print "i ignores and resamples rope"
    print "r removes this entire example"
    print "you can C-c to quit safely"
    response = raw_input("Use this demonstration?[y/N/d/x/i/r]")
    resample = False
    success = False
    if response in ('R', 'r'):
        remove_last_example(outfile)
        resample = True
    elif response in ('I', 'i'):
        resample = True
    elif response in ('D', 'd'):
        resample = True
        # write the demonstration
        write_flush(outfile,
                    items=[['cloud_xyz', start_scene.cloud],
                           ['action', action_name],
                           # additional flag to tell if this is a knot
                           ['knot', 0],
                           ['deadend', 0],
                           ['pred', pred]])
        # write the end state
        write_flush(outfile,
                    items=[['cloud_xyz', next_scene.cloud],
                           ['action', 'endstate:' + action_name],
                           ['knot', 1],
                           ['deadend', 0],
                           ['pred', str(len(outfile) - 1)]])
        success = True
    elif response in ('X', 'x'):
        resample = True
        # write the demonstration
        write_flush(outfile,
                    items=[['cloud_xyz', start_scene.cloud],
                           ['action', action_name],
                           # additional flag to tell if this is a knot
                           ['knot', 0],
                           ['deadend', 0],
                           ['pred', pred]])
        # write the end state
        write_flush(outfile,
                    items=[['cloud_xyz', next_scene.cloud],
                           ['action', 'endstate:' + action_name],
                           ['knot', 0],
                           ['deadend', 1],
                           ['pred', str(len(outfile) - 1)]])

        success = True
    return (success, resample)


def manual_select_demo(args, transferer, sim, lfd_env, outfile, pred):
    scene_state = lfd_env.observe_scene()
    start_state = sim.get_state()
    # ds_clouds = dict(zip(GlobalVars.demos.keys(),
    #                 [d.scene_state.cloud for d in GlobalVars.demos]))

    costs = transferer.registration_factory.batch_cost(scene_state)
    best_keys = sorted(costs, key=costs.get)
    for seg_name in best_keys:
        test_aug_traj = transferer.transfer(GlobalVars.demos[seg_name],
                                            scene_state,
                                            plotting=args.plotting)
        feasible, misgrasp = lfd_env.execute_augmented_trajectory(
            test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
        if not feasible or misgrasp:
            sim.set_state(start_state)
            continue
        new_scene = lfd_env.observe_scene()
        (success, resample) = get_input(scene_state, str(seg_name), new_scene,
                                        outfile, pred)
        if resample or success:
            break
        else:
            sim.set_state(start_state)
    if resample:
        # return the key for the next sample we'll see (so it is its own pred)
        return (str(len(outfile)), resample)
    elif args.label.label_single_step:
        return (str(len(outfile)), True)
    else:
        # return the key for the most recent addition
        return (str(len(outfile) - 1), resample)


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


def load_dagger_state(sampled_state, sim, animation):
    replace_rope(sim, sampled_state['rope_nodes'], animation)
    sim.settle()
    sim.viewer.Step()
    user_input = raw_input(
        "Press i if this loaded state is a knot or deadend, to skip to the next state. Otherwise press enter to continue")
    return user_input not in ['i', 'I']


def load_random_start_segment(demofile):
    start_keys = [seg for seg in GlobalVars.actions.keys() if seg.startswith(
        'demo') and seg.endswith('00')]
    seg_name = random.choice(start_keys)
    return (GlobalVars.actions[seg_name]['cloud_xyz'], seg_name)


def sample_rope_state(rope_args, sim, animation, human_check=False):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        new_xyz, demo_id = load_random_start_segment(GlobalVars.actions)
        perturb_radius = random.uniform(rope_args.min_rad, rope_args.max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud(
            new_xyz,
            perturb_peak_dist=perturb_radius,
            num_perturb_points=rope_args.n_perturb_pts)
        replace_rope(sim, rope_nodes, animation)
        sim.settle()
        sim_util.reset_arms_to_side(sim)

        if animation:
            sim.viewer.Step()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True
    return (rope_nodes, demo_id)


def h5_no_endstate_len(outfile):
    ctr = 0
    for k in outfile:
        if not outfile[k]['knot'][()]:
            ctr += 1
    print "num examples in file:\t", ctr
    return ctr


def write_flush(outfile, items, key=None):
    if not key:
        key = str(len(outfile))
    g = outfile.create_group(key)
    for k, v in items:
        g[k] = v
    outfile.flush()


def remove_last_example(outfile):
    key = str(len(outfile) - 1)
    try:
        while True:
            # will loop until we get something that is its own pred
            new_key = str(outfile[key]['pred'])
            del outfile[key]
            key = new_key
    except:
        key = str(len(outfile) - 1)
        if not outfile[key]['knot']:
            raise Exception("issue deleting examples, check your file")


def check_outfile(outfile, use_dagger=False):
    # Assumes keys in outfile are consecutive integers, starting with 0
    prev_start = 0
    for i in range(len(outfile.keys())):
        k = str(i)
        if not all(sub_g in outfile[k] for sub_g in (
                'action', 'cloud_xyz', 'knot', 'pred')):
            print "missing necessary groups"
            outfile.close()
            return False
        pred = int(outfile[k]['pred'][()])
        # Check that each trajectory has length at least 4 (including endstate)
        if pred == i and i != 0:
            if i - prev_start < 4 and not use_dagger:
                print "trajectory has length less than 4 (including endstate); index: ", k, ", length: ", i - prev_start
                outfile.close()
                return False
            if i - prev_start > 5:
                print "possible mistake: trajectory has length greater than 5 (including endstate); index: ", k, ", length: ", i - prev_start
            if not outfile[str(i - 1)]['knot'][()]:
                if not use_dagger or (
                        use_dagger and not outfile[str(i - 1)]['deadend'][()]):
                    print "trajectory must end with a knot or deadend; index: ", i - 1
                    outfile.close()
                    return False
            prev_start = i
        if pred != int(k) and pred != int(k) - 1:
            print "predecessors not correct", k, pred
            outfile.close()
            return False
        knot = outfile[k]['knot'][()]
        action = outfile[k]['action'][()]
        if knot and not action.startswith('endstate'):
            print "end states labelled improperly"
            outfile.close()
            return False
        if 'deadend' in outfile[k].keys():
            deadend = outfile[k]['deadend'][()]
            if deadend and not action.startswith('deadend'):
                print "deadend states labelled improperly"
                outfile.close()
                return False
    if i - prev_start < 3 and not use_dagger:
        print "trajectory has length less than 4 (including endstate); index: ", k, ", length: ", i - prev_start
        outfile.close()
        return False
    if i - prev_start > 4:
        print "possible mistake: trajectory has length greater than 5 (including endstate); index: ", k, ", length: ", i - prev_start
    if not outfile[str(i)]['knot'][()]:
        if not use_dagger or (
                use_dagger and not outfile[str(i - 1)]['deadend'][()]):
            print "trajectory must end with a knot or deadend; index: ", i - 1
            outfile.close()
            return False

    return True


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
    parser_eval = subparsers.add_parser('label')

    parser_eval.add_argument('actionfile', type=str, nargs='?',
                             default='../bigdata/misc/overhand_actions.h5')
    parser_eval.add_argument('--taskfile', type=str, nargs='?',
                             default='../data/misc/Sep13_r0.1_n7_train.h5')
    parser_eval.add_argument('outfile', type=str, nargs='?')

    parser_eval.add_argument("--n_examples", type=int, default=1000)
    parser_eval.add_argument("--min_rad", type=float, default="0.1",
                             help="min perturbation radius")
    parser_eval.add_argument("--max_rad", type=float, default="0.1",
                             help="max perturbation radius")
    parser_eval.add_argument(
        "--n_perturb_pts",
        type=int,
        default=7,
        help="number of points perturbed from demo start state")
    parser_eval.add_argument("--dagger_states_file", type=str)
    parser_eval.add_argument("--label_single_step", action="store_true")

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
    parser_eval.add_argument("--downsample_size", type=int, default=0.025)
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

    parser_eval.add_argument("--parallel", action="store_true")
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
    GlobalVars.actions = h5py.File(args.label.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.label.actionfile)
    GlobalVars.actions_cache = h5py.File(
        actions_root + '.cache' + actions_ext, 'a')

    GlobalVars.demos = {}
    for action, seg_info in GlobalVars.actions.iteritems():
        if args.label.ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            scene_state = GroundTruthRopeSceneState(
                rope_nodes,
                ROPE_RADIUS,
                upsample=args.label.upsample,
                upsample_rad=args.label.upsample_rad,
                downsample_size=args.label.downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
            scene_state = SceneState(
                full_cloud, downsample_size=args.label.downsample_size)
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
    actions = h5py.File(args.label.actionfile, 'r')

    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(
        actions, args.label.fake_data_segment, args.label.fake_data_transform)
    table_height = init_rope_xyz[:, 2].mean() - .02

    sim_objs = []
    sim_objs.append(
        XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject(
        "table", [1, 0, table_height + (-.1 + .01)], [.85, .85, .1], dynamic=False))

    sim = DynamicSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    if args.label.ground_truth:
        lfd_env = GroundTruthRopeLfdEnvironment(
            sim,
            world,
            upsample=args.label.upsample,
            upsample_rad=args.label.upsample_rad,
            downsample_size=args.label.downsample_size)
    else:
        lfd_env = LfdEnvironment(
            sim, world, downsample_size=args.label.downsample_size)

    dof_inds = sim_util.dof_inds_from_name(
        sim.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(
        *[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    # this also sets the torso (torso_lift_joint) to the height in the data
    sim.robot.SetDOFValues(values, dof_inds)
    sim_util.reset_arms_to_side(sim)

    if args.animation:
        viewer = trajoptpy.GetViewer(sim.env)
        if os.path.isfile(args.window_prop_file) and os.path.isfile(
                args.camera_matrix_file):
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

    if args.label.dof_limits_factor != 1.0:
        assert 0 < args.label.dof_limits_factor and args.label.dof_limits_factor <= 1.0
        active_dof_indices = sim.robot.GetActiveDOFIndices()
        active_dof_limits = sim.robot.GetActiveDOFLimits()
        for lr in 'lr':
            manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
            dof_inds = sim.robot.GetManipulator(manip_name).GetArmIndices()
            limits = np.asarray(sim.robot.GetDOFLimits(dof_inds))
            limits_mean = limits.mean(axis=0)
            limits_width = np.diff(limits, axis=0)
            new_limits = limits_mean + args.label.dof_limits_factor * \
                np.r_[-limits_width / 2.0, limits_width / 2.0]
            for i, ind in enumerate(dof_inds):
                active_dof_limits[0][
                    active_dof_indices.tolist().index(ind)] = new_limits[0, i]
                active_dof_limits[1][
                    active_dof_indices.tolist().index(ind)] = new_limits[1, i]
        sim.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])
    return lfd_env, sim


def setup_registration_and_trajectory_transferer(args, sim):
    if args.label.gpu:
        if args.label.reg_type == 'rpm':
            reg_factory = GpuTpsRpmRegistrationFactory(
                GlobalVars.demos, args.label.actionfile)
        elif args.label.reg_type == 'bij':
            reg_factory = GpuTpsRpmBijRegistrationFactory(
                GlobalVars.demos, args.label.actionfile)
        else:
            raise RuntimeError(
                "Invalid reg_type option %s" % args.label.reg_type)
    else:
        if args.label.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(GlobalVars.demos)
        elif args.label.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
        elif args.label.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(
                GlobalVars.demos, n_iter=10)  # TODO
        else:
            raise RuntimeError(
                "Invalid reg_type option %s" % args.label.reg_type)

    if args.label.transferopt == 'pose' or args.label.transferopt == 'finger':
        traj_transferer = PoseTrajectoryTransferer(
            sim,
            args.label.beta_pos,
            args.label.beta_rot,
            args.label.gamma,
            args.label.use_collision_cost)
        if args.label.transferopt == 'finger':
            traj_transferer = FingerTrajectoryTransferer(
                sim,
                args.label.beta_pos,
                args.label.gamma,
                args.label.use_collision_cost,
                init_trajectory_transferer=traj_transferer)
    else:
        raise RuntimeError("Invalid transferopt option %s" %
                           args.label.transferopt)

    if args.label.unified:
        reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(
            reg_factory, traj_transferer)
    else:
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(
            reg_factory, traj_transferer)
    return reg_and_traj_transferer


def main():
    args = parse_input_args()

#    if args.subparser_name == "eval":
#        eval_util.save_results_args(args.resultfile, args)
#    elif args.subparser_name == "replay":
#        loaded_args = eval_util.load_results_args(args.replay.loadresultfile)
#        assert 'eval' not in vars(args)
#        args.label= loaded_args.label
#    else:
#        raise RuntimeError("Invalid subparser name")

    setup_log_file(args)

    set_global_vars(args)
    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args)
    reg_and_traj_transferer = setup_registration_and_trajectory_transferer(
        args, sim)

    if args.subparser_name == "label":
        label_demos(args, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
