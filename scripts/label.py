#!/usr/bin/env python

from __future__ import division

import os.path
import h5py
import atexit
import random
from string import lower

import trajoptpy
import numpy as np

from lfd.environment import sim_util
from lfd.environment import settings
from lfd.demonstration.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from lfd.environment.simulation import DynamicRopeSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject
from lfd.environment.environment import LfdEnvironment, GroundTruthRopeLfdEnvironment
from lfd.registration.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, BatchGpuTpsRpmBijRegistrationFactory, BatchGpuTpsRpmRegistrationFactory
from lfd.transfer.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from lfd.transfer.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer
from lfd.rapprentice import util
from lfd.rapprentice import task_execution, rope_initialization
from lfd.rapprentice.knot_classifier import isFig8Knot
from lfd.rapprentice.util import redprint





# Usage: python scripts/label.py --animation 2 label bigdata/misc/overhand_actions.h5 data/misc/Sep14_train2.h5 finger bij --gpu

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None

# @profile
def label_demos_parallel(args, transferer, lfd_env, sim):
    outfile = h5py.File(args.eval.outfile, 'a')

    rope_params = sim_util.RopeParams()
    rope_params.radius = settings.ROPE_RADIUS_THICK
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness

    # TODO pass in rope params to sample_rope_state
    (init_rope_nodes, demo_id) = sample_rope_state(
        args.eval, sim, args.animation)
    resample = False



    labeled_data = []


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
        costs = transferer.registration_factory.batch_cost(scene_state)
        best_keys = sorted(costs, key=costs.get)
        for seg_name in best_keys:
            traj = transferer.transfer(GlobalVars.demos[seg_name],
                                                scene_state,
                                                plotting=args.plotting)
        

            feasible, misgrasp = lfd_env.execute_augmented_trajectory(
                traj, step_viewer=args.animation, interactive=args.interactive)
            sim_util.reset_arms_to_side(sim)
            sim.settle(step_viewer=args.animation)

            if not feasible or misgrasp:
                print 'Feasible: ', feasible
                print 'Misgrasp: ', misgrasp
                if misgrasp:
                    sim.set_state(sim_state)
                    continue
            print "y accepts this action"
            print "n rejects this action"
            print "r resamples rope state"
            print "f to save this as a failure"
            print "C-c safely quits"
            user_input = lower(raw_input("What to do?"))
            success = False
            if user_input == 'y':
                success = True
                labeled_data.append((scene_state,seg_name))
                if isFig8Knot(get_rope_nodes(sim)):
                    save_success(outfile, labeled_data)
                    labeled_data = []
                    sim.set_state(sample_rope_state(args.eval, sim))
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
        if not success:
            labeled_data = []
            sim.set_state(sample_rope_state(args.eval, sim))

def save_failure(outfile, failure_scene):
    key = get_next_failure_key(outfile)
    g = outfile.create_group(key)
    g['cloud_xyz'] = failure_scene.cloud
    outfile.flush()

def save_success(outfile, labeled_data):
    i_task = get_next_task_i(outfile)
    print 'Saving ' +str(len(labeled_data)) + 'step knot to results, task', i_task
    for i_step in range(len(labeled_data)):
        scene, action = labeled_data[i_step]
        key = str((i_task, i_step))
        g = outfile.create_group(key)
        g['cloud_xyz'] = scene.cloud
        g['action'] = action
        if i_step == len(labeled_data)-1:
            g['knot'] = 1
        else:
            g['knot'] = 0
    outfile.flush()

def get_rope_nodes(sim):
    for sim_obj in sim.sim_objs:
        if isinstance(sim_obj, RopeSimulationObject):
            rope_sim_obj = sim_obj
            break
    return rope_sim_obj.rope.GetControlPoints()

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

def load_random_start_segment(demofile):
    start_keys = [seg for seg in GlobalVars.actions.keys() if 'seg00' in seg]
    seg_name = random.choice(start_keys)
    return (GlobalVars.actions[seg_name]['cloud_xyz'], seg_name)


def sample_rope_state(rope_args, sim, animation=False, human_check=True):
    print 'Sampling rope state'
    success = False
    while not success:
        new_xyz, demo_id = load_random_start_segment(GlobalVars.actions)
        print demo_id
        perturb_radius = random.uniform(rope_args.min_rad, rope_args.max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud(
            new_xyz,
            perturb_peak_dist=perturb_radius,
            num_perturb_points=rope_args.n_perturb_pts)
        replace_rope(sim, rope_nodes, animation)
        sim.settle()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True
 
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
    parser_eval.add_argument("--downsample_size", type=float, default=0.025)
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
    parser_eval.add_argument("--batch", action="store_true", default=False)

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
                settings.ROPE_RADIUS,
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

    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(
        actions, args.eval.fake_data_segment, args.eval.fake_data_transform)
    table_height = init_rope_xyz[:, 2].mean() - .02

    sim_objs = []
    sim_objs.append(
        XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject(
        "table", [1, 0, table_height + (-.1 + .01)], [.85, .85, .1], dynamic=False))

    print 'Setting up lfd environment'
    sim = DynamicRopeSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    if args.eval.ground_truth:
        lfd_env = GroundTruthRopeLfdEnvironment(
            sim,
            world,
            upsample=args.eval.upsample,
            upsample_rad=args.eval.upsample_rad,
            downsample_size=args.eval.downsample_size)
    else:
        lfd_env = LfdEnvironment(
            sim, world, downsample_size=args.eval.downsample_size)

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

    if args.eval.dof_limits_factor != 1.0:
        assert 0 < args.eval.dof_limits_factor and args.eval.dof_limits_factor <= 1.0
        active_dof_indices = sim.robot.GetActiveDOFIndices()
        active_dof_limits = sim.robot.GetActiveDOFLimits()
        for lr in 'lr':
            manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
            dof_inds = sim.robot.GetManipulator(manip_name).GetArmIndices()
            limits = np.asarray(sim.robot.GetDOFLimits(dof_inds))
            limits_mean = limits.mean(axis=0)
            limits_width = np.diff(limits, axis=0)
            new_limits = limits_mean + args.eval.dof_limits_factor * \
                np.r_[-limits_width / 2.0, limits_width / 2.0]
            for i, ind in enumerate(dof_inds):
                active_dof_limits[0][
                    active_dof_indices.tolist().index(ind)] = new_limits[0, i]
                active_dof_limits[1][
                    active_dof_indices.tolist().index(ind)] = new_limits[1, i]
        sim.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])
    return lfd_env, sim


def setup_registration_and_trajectory_transferer(args, sim):
    print 'Setting up registration'
    if args.eval.batch:
        if args.eval.reg_type == 'rpm':
            reg_factory = BatchGpuTpsRpmRegistrationFactory(
                GlobalVars.demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = BatchGpuTpsRpmBijRegistrationFactory(
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
        label_demos_parallel(args, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()
