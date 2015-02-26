#!/usr/bin/env python

from __future__ import division

import copy
import pprint
import argparse
from core import demonstration, registration, transfer, sim_util
from core.constants import ROPE_RADIUS, MAX_ACTIONS_TO_TRY

from core.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from core.simulation import DynamicSimulationRobotWorld, DynamicRopeSimulationRobotWorld, ClutterSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject, RopeSimulationObject
from core.environment import LfdEnvironment, GroundTruthRopeLfdEnvironment
from core.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory
from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer
from core.action_selection import GreedyActionSelection
from core.action_selection import FeatureActionSelection
from core.transfer_simulate import BatchTransferSimulate

from rapprentice import eval_util, util, task_execution

import pdb, time

from mmqe import search

import trajoptpy, openravepy
from rapprentice.knot_classifier import isKnot as is_knot, isFig8Knot as is_fig8knot, calculateCrossings
import os, os.path, numpy as np, h5py
from rapprentice.util import redprint, yellowprint
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random, hashlib, ipdb, sys

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None
    features = None

def eval_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim):
    """TODO
    
    Args:
        action_selection: ActionSelection
        reg_and_traj_transferer: RegistrationAndTrajectoryTransferer
        lfd_env: LfdEnvironment
        sim: DynamicSimulation
    """
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_indexed_items(holdoutfile, task_list=args.tasks, task_file=args.taskfile, i_start=args.i_start, i_end=args.i_end)

    rope_params = sim_util.RopeParams()
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness


    obj_removed = []
    num_successes = 0
    num_total = 0
    for i_task, task_info in holdout_items:

        redprint("task %s" % i_task)
        init_state = {}
        for k in task_info['state']:
            init_state[k] = task_info['state'][k][:]
        sim.initialize(init_state, step_viewer=args.animation)
        task_start = time.time()
        task_time = 0
        for i_step in range(args.eval.num_steps):
            redprint("task %s step %i" % (i_task, i_step))
            
            sim_util.reset_arms_to_side(sim)
            if args.animation:
                sim.viewer.Step()
            sim_state = sim.get_state()
            sim.set_state(sim_state)
            old_cleared_objs = sim.remove_cleared_objs()
            scene_state = lfd_env.observe_scene()
            # plot cloud of the test scene
            handles = []
            if args.plotting:
                handles.append(sim.env.plot3(scene_state.cloud[:,:3], 2, scene_state.color if scene_state.color is not None else (0,0,1)))
                sim.viewer.Step()

            eval_stats = eval_util.EvalStats()

            start_time = time.time()
            try:
                (agenda, q_values_root), goal_found = action_selection.plan_agenda(scene_state, i_step)
            except ValueError: #e.g. if cloud is empty - any action is hopeless
                redprint("**Raised Value Error during action selection")
                break
            eval_stats.action_elapsed_time += time.time() - start_time
 
            eval_stats.generalized = True
            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1


            for i_choice in range(num_actions_to_try):
                if q_values_root[i_choice] == -np.inf: # none of the demonstrations generalize
                    eval_stats.generalized = False
                    break
                redprint("TRYING %s"%agenda[i_choice])

                best_root_action = str(agenda[i_choice])
                start_time = time.time()
                try:
                    aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], scene_state, plotting=args.plotting)
                except ValueError: # If something is cloud/traj is empty or something
                    redprint("**Raised value error during traj transfer")
                    break
                (eval_stats.feasible, eval_stats.misgrasp, grasped_objs) = lfd_env.execute_augmented_trajectory(
                    aug_traj, step_viewer=args.animation, check_feasible=args.eval.check_feasible, return_grasped_objs=True)
                reward = sim.compute_reward(old_cleared_objs, grasped_objs)

                sim.settle()
                eval_stats.exec_elapsed_time += time.time() - start_time
                
                if not args.eval.check_feasible or eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action and we care about feasibility
                     break
                else:
                     sim.set_state(sim_state)
            print "BEST ACTION:", best_root_action

            sim.remove_cleared_objs()
            objs_left = len(sim.observe_objs) - 1
            success = objs_left <= 0

            results = {'scene_state':scene_state, 
                       'best_action':best_root_action, 
                       'values':q_values_root, 
                       'aug_traj':aug_traj, 
                       'eval_stats':eval_stats, 
                       'sim_state':sim_state, 
                       'success': success, 
                       'objs_left': objs_left}
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, results)
            
            if not eval_stats.generalized:
                assert not knot
                break
            
            if args.eval.check_feasible and not eval_stats.feasible:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                assert not knot
                break
            
            if success:
                num_successes += 1
                break;
        obj_removed.append(5 - objs_left)

        task_time = (.8) * (time.time() - task_start)  + (.2) * task_time if task_time else time.time() - task_start
        num_total += 1
        redprint('Eval Successes / Total: {}/{}'.format(num_successes, num_total))
        redprint('Success Rate: {}'.format(float(num_successes)/num_total))
        redprint('Avg Num Removed: {}'.format(np.mean(obj_removed)))
        redprint('Estimated Time Left: {}'.format((len(holdout_items) - num_total) * task_time))

def replay_on_holdout(args, action_selection, transfer, lfd_env, sim):
    loadresultfile = h5py.File(args.replay.loadresultfile, 'r')
    loadresult_items = eval_util.get_indexed_items(loadresultfile, task_list=args.tasks, task_file=args.taskfile, i_start=args.i_start, i_end=args.i_end)
    
    num_successes = 0
    num_total = 0
    
    for i_task, task_info in loadresult_items:
        redprint("task %s" % i_task)

        for i_step in range(len(task_info)):
            redprint("task %s step %i" % (i_task, i_step))
            
            replay_results = eval_util.load_task_results_step(args.replay.loadresultfile, i_task, i_step)
            sim_state = replay_results['sim_state']

            if i_step > 0: # sanity check for reproducibility
                sim_util.reset_arms_to_side(sim)
                if sim.simulation_state_equal(sim_state, sim.get_state()):
                    yellowprint("Reproducible results OK")
                else:
                    yellowprint("The replayed simulation state doesn't match the one from the result file")
                
            sim.set_state(sim_state)

            if args.replay.simulate_traj_steps is not None and i_step not in args.replay.simulate_traj_steps:
                continue
            
            if i_step in args.replay.compute_traj_steps: # compute the trajectory in this step
                best_root_action = replay_results['best_action']
                scene_state = replay_results['scene_state']
                # plot cloud of the test scene
                handles = []
                if args.plotting:
                    handles.append(sim.env.plot3(scene_state.cloud[:,:3], 2, scene_state.color if scene_state.color is not None else (0,0,1)))
                    sim.viewer.Step()
                test_aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], scene_state, plotting=args.plotting)
            else:
                test_aug_traj = replay_results['aug_traj']
            feasible, misgrasp = lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive, check_feasible=args.eval.check_feasible)
            
            if replay_results['knot']:
                num_successes += 1
        
        num_total += 1
        redprint('REPLAY Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--plotting", type=int, default=1, help="plots if animation != 0 and plotting != 0")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='../.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # arguments for eval
    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str, nargs='?', default='../mmqe_data/mmqe_box_demos_feb12.h5')
    parser_eval.add_argument('holdoutfile', type=str, nargs='?', default='../mmqe_data/cluttered_test.h5')
    parser.add_argument("--landmarkfile", type=str, default='../mmqe_data/landmarks.h5')

    parser_eval.add_argument('action_selection', type=str, nargs='?', choices=['greedy', 'feature'])
    parser_eval.add_argument('--weightfile', type=str, default='')
    parser_eval.add_argument('--feature_type', type=str, nargs='?', choices=['base', 'mul', 'mul_quad', 'mul_quad_ind', 'mul_quad_bendind', 'mul_quad_mapind', 'mul_s', 'mul_grip', 'mul_s_map', 'landmark', 'timestep'], default='base')

    parser_eval.add_argument("transferopt", type=str, nargs='?', choices=['pose', 'finger'], default='finger')
    parser_eval.add_argument("reg_type", type=str, choices=['segment', 'rpm', 'bij'], default='bij')
    parser_eval.add_argument("--unified", type=int, default=0)
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--downsample", type=int, default=1)
    parser_eval.add_argument("--downsample_size", type=float, default=0.025)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    parser_eval.add_argument("--ground_truth", type=int, default=0)
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")

    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--check_feasible", type=int, default=0)

    parser_eval.add_argument("--width", type=int, default=1)
    parser_eval.add_argument("--depth", type=int, default=0)

    parser_eval.add_argument("--alpha", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=100.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--use_collision_cost", type=int, default=1)

    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_param_radius", type=str, default=None)
    parser_eval.add_argument("--rope_param_angStiffness", type=str, default=None)

    parser_eval.add_argument("--use_color", type=int, default=0)


    parser_eval.add_argument("--parallel", action="store_true")
    parser_eval.add_argument("--search_parallel", action="store_true")
    parser_eval.add_argument("--gpu", action="store_true", default=False)

    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--compute_traj_steps", type=int, default=[], nargs='*', metavar='i_step', help="recompute trajectories for the i_step of all tasks")
    parser_replay.add_argument("--simulate_traj_steps", type=int, default=None, nargs='*', metavar='i_step',
                               help="if specified, restore the rope state from file and then simulate for the i_step of all tasks")
                               # if not specified, the rope state is not restored from file, but it is as given by the sequential simulation

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
    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')
    
    GlobalVars.demos = {}
    for action, seg_info in GlobalVars.actions.iteritems():
        if args.eval.ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            scene_state = GroundTruthRopeSceneState(rope_nodes, ROPE_RADIUS, upsample=args.eval.upsample, upsample_rad=args.eval.upsample_rad, downsample_size=args.eval.downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
            scene_state = SceneState(full_cloud, downsample_size=args.eval.downsample_size)
        lr2arm_traj = {}
        lr2finger_traj = {}
        lr2ee_traj = {}
        lr2open_finger_traj = {}
        lr2close_finger_traj = {}
        for lr in 'lr':
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            lr2arm_traj[lr] = np.asarray(seg_info[arm_name])
            lr2finger_traj[lr] = sim_util.gripper_joint2gripper_l_finger_joint_values(np.asarray(seg_info['%s_gripper_joint'%lr]))[:,None]
            lr2ee_traj[lr] = np.asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'])
            lr2open_finger_traj[lr] = np.zeros(len(lr2finger_traj[lr]), dtype=bool)
            lr2close_finger_traj[lr] = np.zeros(len(lr2finger_traj[lr]), dtype=bool)
            opening_inds, closing_inds = sim_util.get_opening_closing_inds(lr2finger_traj[lr])
#             # opening_inds/closing_inds are indices before the opening/closing happens, so increment those indices (if they are not out of bound)
#             opening_inds = np.clip(opening_inds+1, 0, len(lr2finger_traj[lr])-1) # TODO figure out if +1 is necessary
#             closing_inds = np.clip(closing_inds+1, 0, len(lr2finger_traj[lr])-1)
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
            if np.any(closing_inds) and not np.any(opening_inds):
                lr2open_finger_traj[lr][-1] = True


        aug_traj = AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
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
    if args.eval.gpu:
        if args.eval.reg_type == 'rpm':
            reg_factory = GpuTpsRpmRegistrationFactory(GlobalVars.demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = GpuTpsRpmBijRegistrationFactory(GlobalVars.demos, args.eval.actionfile)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(GlobalVars.demos, actionfile=args.eval.actionfile)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)

    if args.eval.transferopt == 'pose' or args.eval.transferopt == 'finger':
        traj_transferer = PoseTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost)
        if args.eval.transferopt == 'finger':
            traj_transferer = FingerTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.gamma, args.eval.use_collision_cost, init_trajectory_transferer=traj_transferer)
    else:
        raise RuntimeError("Invalid transferopt option %s"%args.eval.transferopt)
    
    if args.eval.unified:
        reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    else:
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    return reg_and_traj_transferer

def get_features(args):
    feat_type = args.eval.feature_type
    if feat_type== 'base':
        from mmqe.features import BatchRCFeats as feat
    elif feat_type == 'mul':
        from mmqe.features import MulFeats as feat
    elif feat_type == 'mul_quad':
        from mmqe.features import QuadSimpleMulFeats as feat
    elif feat_type == 'mul_quad_ind':
        from mmqe.features import QuadSimpleMulIndFeats as feat
    elif feat_type == 'mul_quad_mapind':
        from mmqe.features import QuadSimpleMulMapIndFeats as feat
    elif feat_type == 'mul_quad_bendind':
        from mmqe.features import QuadSimpleMulBendIndFeats as feat
    elif feat_type == 'mul_s':
        from mmqe.features import SimpleMulFeats as feat
    elif feat_type == 'mul_grip':
        from mmqe.features import SimpleMulGripperFeats as feat
    elif feat_type == 'mul_s_map':
        from mmqe.features import SimpleMulMapIndFeats as feat
    elif feat_type == 'landmark':
        from mmqe.features import LandmarkFeats as feat
    elif feat_type == 'timestep':
        from mmqe.features import TimestepActionMulFeats as feat
    else:
        raise ValueError('Incorrect Feature Type')

    feats = feat(args.eval.actionfile)
    try:
        feats.set_landmark_file(args.landmarkfile)
    except AttributeError:
        pass
    if args.eval.weightfile:
        feats.load_weights(args.eval.weightfile)
    GlobalVars.features = feats
    return feats


def main():
    args = parse_input_args()

    if args.subparser_name == "eval":
        eval_util.save_results_args(args.resultfile, args)
    elif args.subparser_name == "replay":
        loaded_args = eval_util.load_results_args(args.replay.loadresultfile)
        assert 'eval' not in vars(args)
        args.eval = loaded_args.eval
    else:
        raise RuntimeError("Invalid subparser name")

    setup_log_file(args)

    set_global_vars(args)
    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args)
    reg_and_traj_transferer = setup_registration_and_trajectory_transferer(args, sim)
    if args.eval.action_selection == 'feature':
        get_features(args)
    if args.eval.action_selection == 'greedy':
        action_selection = GreedyActionSelection(reg_and_traj_transferer.registration_factory)
    elif args.eval.search_parallel:
        action_selection = ParallelFeatureActionSelection(reg_and_traj_transferer.registration_factory, GlobalVars.features, GlobalVars.actions, GlobalVars.demos, args=args, lfd_env=lfd_env, width=args.eval.width, depth=args.eval.depth)
    else:
        action_selection = FeatureActionSelection(reg_and_traj_transferer.registration_factory, GlobalVars.features, GlobalVars.actions, GlobalVars.demos, simulator=reg_and_traj_transferer, lfd_env=lfd_env, width=args.eval.width, depth=args.eval.depth, debug=2)

    if args.subparser_name == "eval":
        start = time.time()
        if args.eval.parallel:
            eval_on_holdout_parallel(args, action_selection, lfd_env, sim)
        else:
            eval_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
        print "eval time is:\t{}".format(time.time() - start)
    elif args.subparser_name == "replay":
        replay_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
