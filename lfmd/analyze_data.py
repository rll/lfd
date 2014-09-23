#!/usr/bin/env python

from __future__ import division

import argparse
import sys, os, os.path, numpy as np, h5py
import matplotlib.pyplot as plt

import trajoptpy, openravepy
from rapprentice import util
from rapprentice import math_utils as mu

from core import sim_util
from core.demonstration import SceneState, AugmentedTrajectory, Demonstration
from core.simulation import DynamicSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject
from core.environment import LfdEnvironment
from core.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory
from core.action_selection import GreedyActionSelection

import IPython as ipy

def dof_val_cost(dof_val1, dof_val2, coefs):
    """
    Assumes that the values in dof_val and coefs corresponds to position, rotation, force and torque in that order
    """
    assert len(dof_val1) == len(dof_val2)
    assert len(dof_val1) >= 6
    assert len(dof_val1) <= 12
    cost = 0.0
    if coefs[0] != 0:
        cost = coefs[0] * np.linalg.norm(dof_val2[:3] - dof_val1[:3])
    if coefs[1] != 0:
        rot1 = openravepy.rotationMatrixFromAxisAngle(dof_val1[3:6])
        rot2 = openravepy.rotationMatrixFromAxisAngle(dof_val2[3:6])
        aa_diff = openravepy.axisAngleFromRotationMatrix(rot1.T.dot(rot2))
        cost += coefs[1] * np.linalg.norm(aa_diff)
    if len(dof_val1) > 6 and coefs[2] != 0:
        cost += coefs[2] * np.linalg.norm(dof_val2[6:9] - dof_val1[6:9])
    if len(dof_val1) > 9 and coefs[3] != 0:
        cost += coefs[3] * np.linalg.norm(dof_val2[9:12] - dof_val1[9:12])
    return cost

def dtw(traj1, traj2, dof_cost):
    n = len(traj1)
    m = len(traj2)
    DTW = np.zeros((n+1, m+1))
    DTW[:,0] = np.inf
    DTW[0,:] = np.inf
    DTW[0, 0] = 0
    pointers = np.empty((n+1, m+1), dtype=object)
    pointers.fill(None)
    for t1, dof_val1 in enumerate(traj1):
        #if t1 % 100 == 0: print t1
        t1 = t1+1 # increase by one because we need to 1 index DTW
        for t2, dof_val2 in enumerate(traj2):
            t2 = t2+1 
            best_next = min(DTW[t1-1, t2], DTW[t1, t2-1], DTW[t1-1, t2-1])
            best_next_ind = np.argmin([DTW[t1-1, t2], DTW[t1, t2-1], DTW[t1-1, t2-1]])
            pointers[t1, t2] = [(t1-1,t2), (t1,t2-1), (t1-1,t2-1)][best_next_ind]
            DTW[t1, t2] = dof_cost(dof_val1, dof_val2) + best_next
    return DTW, pointers

def align_trajs(trajs, dof_cost):
    aligned_trajs = []
    for i_choice, traj in enumerate(trajs):
        if i_choice == 0:
            aligned_trajs.append(traj)
        else:
            DTW, pointers = dtw(trajs[0], traj, dof_cost)
            aligned_traj = [[] for _ in range(len(trajs[0]))]
            next_ij = (len(trajs[0]), len(traj))
            while next_ij[0] > 0 and next_ij[0] > 0:
                (i,j) = next_ij
                aligned_traj[i-1].append(traj[j-1])
                next_ij = pointers[i,j]
            aligned_traj = [dof_vals[len(dof_vals)//2] for dof_vals in aligned_traj] # pick one in the middle
            aligned_trajs.append(np.asarray(aligned_traj))
    aligned_trajs = np.asarray(aligned_trajs)
    return aligned_trajs

def flipped_angle_axis(aa_trajs):
    angle_trajs = []
    axis_trajs = []
    for aa_traj in aa_trajs:
        assert aa_traj.shape[1] == 3
        angle_traj = np.apply_along_axis(np.linalg.norm, 1, aa_traj)
        axis_traj = aa_traj / angle_traj[:,None]
        angle_trajs.append(angle_traj)
        axis_trajs.append(axis_traj)
    # flip axes of first hmat for all the trajectories
    for axis_traj, angle_traj in zip(axis_trajs[1:], angle_trajs[1:]):
        if axis_traj[0].dot(axis_trajs[0][0]) < 0:
            axis_traj[0] *= -1
            angle_traj[0] = -1 * (angle_traj[0] % 2*np.pi) + 2*np.pi
    # flip axes for the rest of the trajectory for each trajectory
    for axis_traj, angle_traj in zip(axis_trajs, angle_trajs):
        dot_products = np.einsum('ij,ij->i', axis_traj[:-1], axis_traj[1:]) # pairwise dot products axis_traj[t].dot(axis_traj[t-1])
        flip_inds = np.r_[False, dot_products < 0]
        for flip_ind in  np.where(flip_inds)[0]:
            if flip_ind != len(flip_inds)-1:
                flip_inds[flip_ind+1:] = np.invert(flip_inds[flip_ind+1:])
        axis_traj[flip_inds] *= -1
        angle_traj[flip_inds] = -1 * (angle_traj[flip_inds] % 2*np.pi) + 2*np.pi
    # make all angles be between 0 and 2*np.pi
    for angle_traj in angle_trajs:
        angle_traj[:] = np.unwrap(angle_traj)
    return angle_trajs, axis_trajs

def flip_angle_axis_in_place(aa_trajs):
    angle_trajs, axis_trajs = flipped_angle_axis(aa_trajs)
    for aa_traj, axis_traj, angle_traj in zip(aa_trajs, axis_trajs, angle_trajs):
        aa_traj[:,:] = axis_traj * angle_traj[:,None]
    return aa_trajs
        
def analyze_data(args, demos, action_selection, reg_factory, lfd_env, sim):
    # for now, use the scene state of the first demo as the current scene state
    _, demo = demos.items()[0]
    scene_state = demo.scene_state

    handles = []
    handles.append(sim.env.plot3(scene_state.full_cloud, 2, (0,0,1)))
    sim.viewer.Step()

    sys.stdout.write("registering all scenes... ")
    sys.stdout.flush()
    agenda, q_values_root = action_selection.plan_agenda(scene_state)
    print "done"
    
    lr = 'r'
    max_num_demos = min(args.eval.max_num_demos, len(demos))

    ee_trajs = []
    trajs = []
    n_dof = 6
    for i_choice in range(max_num_demos):
        action = agenda[i_choice]
        demo = demos[action]
        reg = reg_factory.register(demo, scene_state)
        ee_traj = reg.f.transform_hmats(demo.aug_traj.lr2ee_traj[lr])
        ee_trajs.append(ee_traj)
        aas = np.empty((len(ee_traj), 3)) # angle axis rotations
        T_x = np.eye(4) # transformation that rotates by pi around the x axis
        T_x[:3,:3] = openravepy.rotationMatrixFromAxisAngle(np.r_[1, 0, 0] * np.pi)
        flip = False # rotate ee_traj by T_x?
        for t, hmat in enumerate(ee_traj):
            if i_choice > 0 and t == 0:
                hmat0 = ee_trajs[0][0]
                angle_diff = np.linalg.norm(openravepy.axisAngleFromRotationMatrix(mu.invertHmat(hmat0).dot(hmat)))
                angle_diff_flipped = np.linalg.norm(openravepy.axisAngleFromRotationMatrix(mu.invertHmat(hmat0).dot(hmat.dot(T_x))))
                flip = angle_diff_flipped < angle_diff
            if flip:
                hmat = hmat.dot(T_x)
            aas[t,:] = openravepy.axisAngleFromRotationMatrix(hmat)
            assert 0 <= np.linalg.norm(aas[t,3:]) and np.linalg.norm(aas[t,3:]) <= 2*np.pi
        force_traj = np.zeros((len(ee_traj),0))
        if demo.aug_traj.lr2force_traj:
            if i_choice == 0:
                n_dof += 3
            force_traj = reg.f.transform_vectors(ee_traj[:,:3,3], demo.aug_traj.lr2force_traj[lr])
        torque_traj = np.zeros((len(ee_traj),0))
        if demo.aug_traj.lr2torque_traj:
            if i_choice == 0:
                n_dof += 3
            torque_traj = reg.f.transform_vectors(ee_traj[:,:3,3], demo.aug_traj.lr2torque_traj[lr])
        traj = np.c_[ee_traj[:,:3,3], aas, force_traj, torque_traj]
        trajs.append(traj)
    
    for traj in trajs:
        handles.append(sim.env.drawlinestrip(traj[:,:3], 2, np.r_[np.random.random(3),1]))
    sim.viewer.Step()
    
    flip_angle_axis_in_place([traj[:,3:6] for traj in trajs])

    plt.ion()
    fig = plt.figure()
    for i_coord in range(n_dof):
        plt.subplot(n_dof,1,i_coord+1)
        for traj in trajs:
            plt.plot(traj[:,i_coord])
    plt.draw()

    sys.stdout.write("aligning trajectories... ")
    sys.stdout.flush()
    coefs = np.r_[args.eval.pos_coef, args.eval.rot_coef, args.eval.force_coef, args.eval.torque_coef]
    aligned_trajs = align_trajs(trajs, lambda dof_val1, dof_val2: dof_val_cost(dof_val1, dof_val2, coefs))
    print "done"
    t_steps = aligned_trajs.shape[1]
    
    fig = plt.figure()
    for i_coord in range(n_dof):
        plt.subplot(n_dof,1,i_coord+1)
        for traj in aligned_trajs:
            plt.plot(traj[:,i_coord])
    plt.draw()

    for t in range(t_steps):
        dof_val = aligned_trajs[:,t,:]
        dof_val_mu = dof_val.mean(axis=0)
        dof_val_sigma = (dof_val-dof_val_mu).T.dot(dof_val-dof_val_mu)/dof_val.shape[0]
        
        pos = aligned_trajs[:,t,:3]
        pos_mu = pos.mean(axis=0)
        pos_sigma = (pos-pos_mu).T.dot(pos-pos_mu)/pos.shape[0]
        U, s, V = np.linalg.svd(pos_sigma)
        T = np.eye(4)
        T[:3,:3] = args.eval.std_dev * U * np.sqrt(s)
        T[:3,3] = pos_mu
        handles.append(sim.viewer.PlotEllipsoid(T, (0,1,0,1), True))

        rot = aligned_trajs[:,t,3:6]
        rot_mu = rot.mean(axis=0)
        rot_sigma = (rot-rot_mu).T.dot(rot-rot_mu)/rot.shape[0]
        U, s, V = np.linalg.svd(rot_sigma)
        rt_sigma_rot = (U * np.sqrt(s)).dot(U.T)
        rot_sigma_pts = [rot_mu]
        for i in range(3):
            rot_sigma_pts.append(rot_mu + args.eval.std_dev * rt_sigma_rot[:,i])
            rot_sigma_pts.append(rot_mu - args.eval.std_dev * rt_sigma_rot[:,i])
        for sigma_pt in rot_sigma_pts:
            hmat = np.eye(4)
            hmat[:3,:3] = openravepy.rotationMatrixFromAxisAngle(sigma_pt)
            hmat[:3,3] = pos_mu
            handles.extend(sim_util.draw_axis(sim, hmat, arrow_length=.01, arrow_width=.001))

    sim.viewer.Step()
    sim.viewer.Idle()
    ipy.embed()

class ForceAugmentedTrajectory(AugmentedTrajectory):
    def __init__(self, lr2force_traj, lr2torque_traj, lr2arm_traj=None, lr2finger_traj=None, lr2ee_traj=None, lr2open_finger_traj=None, lr2close_finger_traj=None):
        super(ForceAugmentedTrajectory, self).__init__(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, 
                                                       lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        self.lr2force_traj = lr2force_traj
        self.lr2torque_traj = lr2torque_traj
    
    def get_resampled_traj(self, timesteps_rs):
        aug_traj = super(ForceAugmentedTrajectory, self).get_resampled_traj(timesteps_rs)
        lr2force_traj_rs = None if self.lr2force_traj is None else {}
        lr2torque_traj_rs = None if self.lr2torque_traj is None else {}
        for (lr2traj_rs, self_lr2traj) in [(lr2force_traj_rs, self.lr2force_traj), (lr2torque_traj_rs, self.lr2torque_traj)]:
            if self_lr2traj is None:
                continue
            for lr in self_lr2traj.keys():
                lr2traj_rs[lr] = mu.interp2d(timesteps_rs, np.arange(len(self_lr2traj[lr])), self_lr2traj[lr])
        return ForceAugmentedTrajectory(lr2force_traj_rs, lr2torque_traj_rs, lr2arm_traj=aug_traj.lr2arm_traj, lr2finger_traj=aug_traj.lr2finger_traj, lr2ee_traj=aug_traj.lr2ee_traj, 
                                 lr2open_finger_traj=aug_traj.lr2open_finger_traj, lr2close_finger_traj=aug_traj.lr2close_finger_traj)

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='../.win_prop.txt')

    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str)
    parser_eval.add_argument("reg_type", type=str, choices=['segment', 'rpm', 'bij'], default='bij')
    parser_eval.add_argument("--downsample_size", type=float, default=0.025)
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser_eval.add_argument("--gpu", action="store_true", default=False)

    parser_eval.add_argument("--pos_coef", type=float, default=1, help="coefficient for dtw position cost")
    parser_eval.add_argument("--rot_coef", type=float, default=.1, help="coefficient for dtw rotation cost")
    parser_eval.add_argument("--force_coef", type=float, default=1, help="coefficient for dtw force cost")
    parser_eval.add_argument("--torque_coef", type=float, default=1, help="coefficient for dtw torque cost")

    parser_eval.add_argument("--downsample_traj", type=int, default=1, help="downsample demonstration trajectory by this factor")
    parser_eval.add_argument("--std_dev", type=float, default=1, help="number of standard deviations plotted for the covariance")
    parser_eval.add_argument("--max_num_demos", type=int, default=10, help="maximum number of demos to combine")

    args = parser.parse_args()
    return args

def setup_demos(args):
    actions = h5py.File(args.eval.actionfile, 'r')
    
    demos = {}
    for action, seg_info in actions.iteritems():
        if 'overhand' in args.eval.actionfile and 'seg00' not in action: continue #TODO
        full_cloud = seg_info['cloud_xyz'][()]
        scene_state = SceneState(full_cloud, downsample_size=args.eval.downsample_size)
        lr2force_traj = {}
        lr2torque_traj = {}
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
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
            if 'end_effector_forces' in seg_info:
                lr2force_traj[lr] = np.asarray(seg_info['end_effector_forces'])[:,:3,0]
                lr2torque_traj[lr] = np.asarray(seg_info['end_effector_forces'])[:,3:,0]
        aug_traj = ForceAugmentedTrajectory(lr2force_traj, lr2torque_traj, lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        if args.eval.downsample_traj > 1:
            aug_traj = aug_traj.get_resampled_traj(np.arange(aug_traj.n_steps)[::args.eval.downsample_traj])
        demo = Demonstration(action, scene_state, aug_traj)
        demos[action] = demo
    return demos

def setup_lfd_environment_sim(args, demos):
    actions = h5py.File(args.eval.actionfile, 'r')
        
    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, args.eval.fake_data_segment, args.eval.fake_data_transform) 
    table_height = init_rope_xyz[:,2].min()

    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height -.1], [.85, .85, .1], dynamic=False))
    
    sim = DynamicSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    lfd_env = LfdEnvironment(sim, world, downsample_size=args.eval.downsample_size)
    
    actions = h5py.File(args.eval.actionfile, 'r')
    _, seg_info = actions.items()[0]
    init_joint_names = np.asarray(seg_info["joint_states"]["name"])
    init_joint_values = np.asarray(seg_info["joint_states"]["position"][0])
    dof_inds = sim_util.dof_inds_from_name(sim.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    sim.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
    sim_util.reset_arms_to_side(sim)
    
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

def setup_registration(args, demos, sim):
    if args.eval.gpu:
        if args.eval.reg_type == 'rpm':
            reg_factory = GpuTpsRpmRegistrationFactory(demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = GpuTpsRpmBijRegistrationFactory(demos, args.eval.actionfile)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(demos)
        elif args.eval.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(demos, n_iter=10) #TODO
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)

    return reg_factory

def main():
    args = parse_input_args()

    demos = setup_demos(args)
    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args, demos)
    reg_factory = setup_registration(args, demos, sim)
    action_selection = GreedyActionSelection(reg_factory)
    analyze_data(args, demos, action_selection, reg_factory, lfd_env, sim)

if __name__ == "__main__":
    main()
