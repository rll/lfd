from __future__ import division

import time
import os
import os.path
import h5py
import atexit

import trajoptpy
import numpy as np
import IPython as ipy

from lfd.environment import sim_util
from lfd.constants import ROPE_RADIUS
from lfd.demonstration.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject, RopeSimulationObject
from lfd.environment.environment import RecordingSimulationEnvironment
from lfd.rapprentice import eval_util
from lfd.registration.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory
from lfd.registration.registration_gpu import BatchGpuTpsRpmBijRegistrationFactory, BatchGpuTpsRpmRegistrationFactory
from lfd.transfer.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from lfd.transfer.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer
from lfd.action_selection import GreedyActionSelection
from lfd.rapprentice import util
from lfd.rapprentice import task_execution
from lfd.rapprentice.knot_classifier import isKnot as is_knot
from lfd.rapprentice.util import redprint


class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--plotting", type=int, default=1, help="plots if animation != 0 and plotting != 0")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")

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

    parser_eval = subparsers.add_parser('eval')

    parser_eval.add_argument("actionfile", type=str, nargs='?', default='data/misc/actions.h5')
    parser_eval.add_argument("holdoutfile", type=str, nargs='?', default='data/misc/holdout_set.h5')
    parser_eval.add_argument("savefile", type=str, default="data/saves/s.h5", help="h5 file in which to store recorded trajectories, roeps states, etc.")

    parser_eval.add_argument("--transferopt", type=str, nargs='?', choices=['pose', 'finger'], default='finger')
    parser_eval.add_argument("--reg_type", type=str, choices=['segment', 'rpm', 'bij'], default='bij')
    parser_eval.add_argument("--unified", type=int, default=0)
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelves', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--downsample_size", type=int, default=0.025)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    parser_eval.add_argument("--ground_truth", type=int, default=1)
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    
    parser_eval.add_argument("--search_until_feasible", action="store_true")

    parser_eval.add_argument("--alpha", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=100.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--use_collision_cost", type=int, default=1)

    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_param_radius", type=str, default=None)
    parser_eval.add_argument("--rope_param_angStiffness", type=str, default=None)
    
    parser_eval.add_argument("--parallel", action="store_true")
    parser_eval.add_argument("--batch", action="store_true")

    args = parser.parse_args()
    if not args.animation:
        args.plotting = 0
    return args

def run_pid(args, action_selection, reg_and_traj_transferer, lfd_env):
    pass

def run_and_record(args, action_selection, reg_and_traj_transferer, lfd_env):

    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_holdout_items(holdoutfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    rope_params = sim_util.RopeParams()
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        redprint("task %s" % i_task)
        sim_util.reset_arms_to_side(lfd_env)
        init_rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        rope = RopeSimulationObject("rope", init_rope_nodes, rope_params)

        lfd_env.add_object(rope)
        lfd_env.settle(step_viewer=args.animation)
        
        next_state = lfd_env.observe_scene()
        
        if args.animation:
            lfd_env.viewer.Step()

        for i_step in range(args.eval.num_steps):
            redprint("task %s step %i" % (i_task, i_step))
            
            state = next_state
            orig_rope_nodes = rope.get_bullet_objects()[0].GetNodes()

            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1
            eval_stats = eval_util.EvalStats()

            try:
                agenda, q_values_root = action_selection.plan_agenda(next_state)
            except ValueError: #e.g. if cloud is empty - any action is hopeless
                break

            unable_to_generalize = False
            for i_choice in range(num_actions_to_try):
                if q_values_root[i_choice] == -np.inf: # none of the demonstrations generalize
                    unable_to_generalize = True
                    break
                redprint("TRYING %s"%agenda[i_choice])

                best_root_action = agenda[i_choice]

                start_time = time.time()
                test_aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], state, plotting=args.plotting)
                lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
                sim_util.reset_arms_to_side(lfd_env)
                if args.animation:
                    lfd_env.viewer.Step()
                # TODO
                eval_stats.success = True
                eval_stats.feasible = True
                eval_stats.misgrasp = False
                next_state = lfd_env.observe_scene()
                eval_stats.exec_elapsed_time += time.time() - start_time

                if eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action
                     break
            if unable_to_generalize:
                 break
            print "BEST ACTION:", best_root_action

            if not eval_stats.feasible:  # If not feasible, restore state
                next_state = state
            
            try:
                save_recorded_traj(args, lfd_env.robot, state, next_state, int(i_task), i_step)
            except Exception as err:
                print err
                ipy.embed()

            if not eval_stats.feasible:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                break
            
            if is_knot(rope.rope.GetControlPoints()):
                num_successes += 1
                break;

        lfd_env.remove_object(rope)
        
        num_total += 1
        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))


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
            # opening_inds/closing_inds are indices before the opening/closing happens, so increment those indices (if they are not out of bound)
            opening_inds = np.clip(opening_inds+1, 0, len(lr2finger_traj[lr])-1) # TODO figure out if +1 is necessary
            closing_inds = np.clip(closing_inds+1, 0, len(lr2finger_traj[lr])-1)
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
        aug_traj = AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        demo = Demonstration(action, scene_state, aug_traj)
        GlobalVars.demos[action] = demo

def save_recorded_traj(args, robot, start_state, end_state, demo_ind, seg_ind):
    print "saving recorded traj"
    if os.path.exists(args.eval.savefile):
        hdf = h5py.File(args.eval.savefile, 'a')
    else:
        hdf = h5py.File(args.eval.savefile, 'w')
    demo_name = "demo%02i-seg%02i"%(demo_ind, seg_ind)
    hdf.create_group(demo_name)
    hdf[demo_name].create_dataset("rope_nodes", data=start_state.rope_nodes)
    hdf[demo_name].create_dataset("cloud", data=start_state.rope_nodes)
    
    # hdf[demo_name].create_group("joint_states")
    # joint_names = np.zeros(5) #in recording these come from the bag file; check out record_demo to see where it gets them
    # traj = np.zeros(5) #ditto; how to format this?
    # hdf[demo_name]["joint_states"].create_dataset("name", data=joint_names)
    # hdf[demo_name]["joint_states"].create_dataset("position", data=traj)

    for manip_name in ["leftarm", "rightarm"]:
        manip = robot.GetManipulator(manip_name)
        joints = np.array([step.manip_trajs[manip_name][0] for step in end_state.history])
        hdf[demo_name].create_dataset(manip_name, data=joints)
        linkname = {"leftarm":"l_gripper_tool_frame", "rightarm":"r_gripper_tool_frame"}[manip_name]
        tf_link = robot.GetLink(linkname)
        hmats = []
        for step in end_state.history:
            joint_vals, dof_inds = step.manip_trajs[manip_name]
            robot.SetDOFValues(joint_vals, dof_inds)
            hmats.append(tf_link.GetTransform())
        hdf[demo_name].create_group(linkname)
        hdf[demo_name][linkname]["hmat"] = np.array(hmats) #"hmat" is unnecessary, kept for backwards compatibility. TODO: just make linkname a dataset
    for lr in 'lr':
        joint_vals = []
        for step in end_state.history:
            joint_vals.append(step.gripper_vals["%s_gripper_l_finger_joint"%lr]) #why is it the l_finger joint?
        hdf[demo_name].create_dataset("%s_gripper_joint"%lr, data=np.array(joint_vals))
    hdf.close()

def setup_lfd_environment(args):
    actions = h5py.File(args.eval.actionfile, 'r')
    
    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, args.eval.fake_data_segment, args.eval.fake_data_transform) 
    table_height = init_rope_xyz[:,2].mean() - .02
    
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height + (-.1 + .01)], [.85, .85, .1], dynamic=False))
    if 'bookshelves' in args.eval.obstacles:
        sim_objs.append(XmlSimulationObject("../data/bookshelves.env.xml", dynamic=False))
    if 'boxes' in args.eval.obstacles:
        sim_objs.append(BoxSimulationObject("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12], dynamic=False))
        sim_objs.append(BoxSimulationObject("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08], dynamic=False))
    if 'cylinders' in args.eval.obstacles:
        sim_objs.append(CylinderSimulationObject("cylinder0", [.7,.43,table_height+(.01+.5)], .12, 1., dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder1", [.7,-.43,table_height+(.01+.5)], .12, 1., dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder2", [.4,.2,table_height+(.01+.65)], .06, .5, dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder3", [.4,-.2,table_height+(.01+.65)], .06, .5, dynamic=False))
    
    lfd_env = RecordingSimulationEnvironment(sim_objs, downsample_size=args.eval.downsample_size)

    dof_inds = sim_util.dof_inds_from_name(lfd_env.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    lfd_env.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
    sim_util.reset_arms_to_side(lfd_env)
    
    if args.animation:
        lfd_env.viewer = trajoptpy.GetViewer(lfd_env.env)
        if os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                lfd_env.viewer.SetWindowProp(*window_prop)
                lfd_env.viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            lfd_env.viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = lfd_env.viewer.GetWindowProp()
                camera_matrix = lfd_env.viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        lfd_env.viewer.Step()
    
    if args.eval.dof_limits_factor != 1.0:
        assert 0 < args.eval.dof_limits_factor and args.eval.dof_limits_factor <= 1.0
        active_dof_indices = lfd_env.robot.GetActiveDOFIndices()
        active_dof_limits = lfd_env.robot.GetActiveDOFLimits()
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            dof_inds = lfd_env.robot.GetManipulator(manip_name).GetArmIndices()
            limits = np.asarray(lfd_env.robot.GetDOFLimits(dof_inds))
            limits_mean = limits.mean(axis=0)
            limits_width = np.diff(limits, axis=0)
            new_limits = limits_mean + args.eval.dof_limits_factor * np.r_[-limits_width/2.0, limits_width/2.0]
            for i, ind in enumerate(dof_inds):
                active_dof_limits[0][active_dof_indices.tolist().index(ind)] = new_limits[0,i]
                active_dof_limits[1][active_dof_indices.tolist().index(ind)] = new_limits[1,i]
        lfd_env.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])
    return lfd_env

def setup_registration_and_trajectory_transferer(args, lfd_env):
    if args.eval.batch:
        if args.eval.reg_type == 'rpm':
            reg_factory = BatchGpuTpsRpmRegistrationFactory(GlobalVars.demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = BatchGpuTpsRpmBijRegistrationFactory(GlobalVars.demos, args.eval.actionfile) # TODO remove n_iter
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(GlobalVars.demos) # TODO remove n_iter        
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)

    if args.eval.transferopt == 'pose' or args.eval.transferopt == 'finger':
        traj_transferer = PoseTrajectoryTransferer(lfd_env, args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost)
        if args.eval.transferopt == 'finger':
            traj_transferer = FingerTrajectoryTransferer(lfd_env, args.eval.beta_pos, args.eval.gamma, args.eval.use_collision_cost, init_trajectory_transferer=traj_transferer)
    else:
        raise RuntimeError("Invalid transferopt option %s"%args.eval.transferopt)
    
    if args.eval.unified:
        reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    else:
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    return reg_and_traj_transferer

def main():
    args = parse_input_args()  
    setup_log_file(args)  
    set_global_vars(args)
    trajoptpy.SetInteractive(args.interactive)
    lfd_env = setup_lfd_environment(args)
    reg_and_traj_transferer = setup_registration_and_trajectory_transferer(args, lfd_env)
    action_selection = GreedyActionSelection(reg_and_traj_transferer.registration_factory)

    if args.subparser_name == "eval":
        start = time.time()
        run_and_record(args, action_selection, reg_and_traj_transferer, lfd_env)
        print "run time is:\t{}".format(time.time() - start)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()