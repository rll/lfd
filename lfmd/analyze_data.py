#!/usr/bin/env python

from __future__ import division

import argparse
import sys, os, os.path, numpy as np, h5py
import matplotlib.pyplot as plt
import scipy
from scipy import optimize

import trajoptpy, openravepy
from rapprentice import util
from rapprentice import math_utils, planning

from core import sim_util, demonstration
from core.demonstration import SceneState, AugmentedTrajectory, Demonstration
from core.simulation import DynamicSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject
from core.environment import LfdEnvironment
from core.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory
from core.action_selection import GreedyActionSelection
from core.transfer import TrajectoryTransferer

import IPython as ipy

def dof_val_cost(dof_val1, dof_val2, coefs):
    """
    Assumes that the values in dof_val and coefs corresponds to position, rotation, position velocity, rotation velocity, force and torque in that order
    """
    assert len(dof_val1) == len(dof_val2)
    assert len(dof_val1) >= 12
    assert len(dof_val1) <= 18
    cost = 0.0
    if coefs[0] != 0:
        cost = coefs[0] * np.linalg.norm(dof_val2[:3] - dof_val1[:3])
    if coefs[1] != 0:
        rot1 = openravepy.rotationMatrixFromAxisAngle(dof_val1[3:6])
        rot2 = openravepy.rotationMatrixFromAxisAngle(dof_val2[3:6])
        aa_diff = openravepy.axisAngleFromRotationMatrix(rot2.dot(rot1.T))
        cost += coefs[1] * np.linalg.norm(aa_diff)
    if coefs[2] != 0:
        cost = coefs[2] * np.linalg.norm(dof_val2[6:9] - dof_val1[6:9])
    if coefs[3] != 0:
        rot_vel1 = openravepy.rotationMatrixFromAxisAngle(dof_val1[9:12])
        rot_vel2 = openravepy.rotationMatrixFromAxisAngle(dof_val2[9:12])
        aa_diff = openravepy.axisAngleFromRotationMatrix(rot_vel2.dot(rot_vel1.T))
        cost += coefs[3] * np.linalg.norm(aa_diff)
    if len(dof_val1) > 12 and coefs[4] != 0:
        cost += coefs[4] * np.linalg.norm(dof_val2[12:15] - dof_val1[12:15])
    if len(dof_val1) > 15 and coefs[5] != 0:
        cost += coefs[5] * np.linalg.norm(dof_val2[15:18] - dof_val1[15:18])
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
    t1_pen = np.zeros((n+1, m+1))
    t2_pen = np.zeros((n+1, m+1))
    for t1, dof_val1 in enumerate(traj1):
        #if t1 % 100 == 0: print t1
        t1 = t1+1 # increase by one because we need to 1 index DTW
        for t2, dof_val2 in enumerate(traj2):
            t2 = t2+1 
            best_next = min(DTW[t1-1, t2], DTW[t1, t2-1], DTW[t1-1, t2-1])
            best_next_ind = np.argmin([DTW[t1-1, t2], DTW[t1, t2-1], DTW[t1-1, t2-1]])
            pointers[t1, t2] = [(t1-1,t2), (t1,t2-1), (t1-1,t2-1)][best_next_ind]
            if best_next_ind == 0: #t2 is constant
                t1_pen[t1, t2] = 0
                t2_pen[t1, t2] = t2_pen[t1-1, t2] + 1
            elif best_next_ind == 1: #t1 is constant
                t1_pen[t1, t2] = t1_pen[t1, t2-1] + 1
                t2_pen[t1, t2] = 0
            else:
                t1_pen[t1, t2] = 0
                t2_pen[t1, t2] = 0
            DTW[t1, t2] = dof_cost(dof_val1, dof_val2) + best_next# + t1_pen[t1, t2] + t2_pen[t1, t2]
    return DTW, pointers, t1_pen, t2_pen

def align_trajs(trajs, dof_cost):
    trajs_timesteps_rs = []
    for i_choice, traj in enumerate(trajs):
        if i_choice == 0:
            trajs_timesteps_rs.append(range(len(trajs[0])))
        else:
            t1_pens = []
            t2_pens = []
            DTW, pointers, t1_pen, t2_pen = dtw(trajs[0], traj, dof_cost)
            traj_timesteps_rs = [[] for _ in range(len(trajs[0]))]
            next_ij = (len(trajs[0]), len(traj))
            while next_ij[0] > 0 and next_ij[0] > 0:
                (i,j) = next_ij
                traj_timesteps_rs[i-1].append(j-1)
                next_ij = pointers[i,j]
                t1_pens.append(t1_pen[i,j])
                t2_pens.append(t2_pen[i,j])
            traj_timesteps_rs = [np.mean(j) for j in traj_timesteps_rs]
            trajs_timesteps_rs.append(np.asarray(traj_timesteps_rs))
#             print np.max(t1_pens), np.max(t2_pens)
    trajs_timesteps_rs = np.asarray(trajs_timesteps_rs)
    return trajs_timesteps_rs

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

def transform_aug_trajs(lr, regs, aug_trajs, flip_rots=True):
    """
    flip_rots: whether to treat 180 deg gripper rotations as the same and this flip these rotations so that the change is minimal
    """
    ee_trajs = []
    trajs = []
    if flip_rots:
        T_x = np.eye(4) # transformation that rotates by pi around the x axis
        T_x[:3,:3] = openravepy.rotationMatrixFromAxisAngle(np.r_[1, 0, 0] * np.pi)
    for i_choice, (reg, aug_traj) in enumerate(zip(regs, aug_trajs)):
        ee_traj = reg.f.transform_hmats(aug_traj.lr2ee_traj[lr])
        ee_trajs.append(ee_traj)
        aa_traj = np.empty((len(ee_traj), 3)) # angle axis rotations
        for t, hmat in enumerate(ee_traj):
            if flip_rots and i_choice > 0:
                if t == 0:
                    hmat0 = ee_trajs[0][0]
                    angle_diff = np.linalg.norm(openravepy.axisAngleFromRotationMatrix(hmat[:3,:3].dot(hmat0[:3,:3].T)))
                    angle_diff_flipped = np.linalg.norm(openravepy.axisAngleFromRotationMatrix((hmat.dot(T_x)[:3,:3]).dot(hmat0[:3,:3].T)))
                    flip_rot = angle_diff_flipped < angle_diff
                if flip_rot: # rotate ee_traj by T_x
                    ee_traj[t] = hmat.dot(T_x)
            aa_traj[t,:] = openravepy.axisAngleFromRotationMatrix(hmat)
            assert 0 <= np.linalg.norm(aa_traj[t,3:]) and np.linalg.norm(aa_traj[t,3:]) <= 2*np.pi
        dt = 0.01
        vel_traj = np.diff(ee_traj[:,:3,3], axis=0)
        vel_traj = np.r_[vel_traj[0][None,:], vel_traj]
        vel_traj /= dt
        aa_vel_traj = np.empty((len(ee_traj)-1, 3))
        for t in range(len(ee_traj)-1):
            rot_diff = ee_traj[t+1,:3,:3].dot(ee_traj[t,:3,:3].T)
            aa_vel_traj[t,:] = openravepy.axisAngleFromRotationMatrix(rot_diff)
        aa_vel_traj = np.r_[aa_vel_traj[0][None,:], aa_vel_traj]
        aa_vel_traj /= dt
        force_traj = np.zeros((len(ee_traj),0))
        if aug_traj.lr2force_traj:
            force_traj = reg.f.transform_vectors(ee_traj[:,:3,3], aug_traj.lr2force_traj[lr])
        torque_traj = np.zeros((len(ee_traj),0))
        if aug_traj.lr2torque_traj:
            torque_traj = reg.f.transform_vectors(ee_traj[:,:3,3], aug_traj.lr2torque_traj[lr])
        traj = np.c_[ee_traj[:,:3,3], aa_traj, vel_traj, aa_vel_traj, force_traj, torque_traj]
        trajs.append(traj)
    return trajs

class MultipleDemosPoseTrajectoryTransferer(TrajectoryTransferer):
    def __init__(self, sim, pos_coef, rot_coef, pos_vel_coef, rot_vel_coef, force_coef, torque_coef, 
                 beta_pos, beta_rot, gamma, use_collision_cost, 
                 downsample_traj=1):
        super(MultipleDemosPoseTrajectoryTransferer, self).__init__(sim, beta_pos, gamma, use_collision_cost)
        self.pos_coef = pos_coef
        self.rot_coef = rot_coef
        self.pos_vel_coef = pos_vel_coef
        self.rot_vel_coef = rot_vel_coef
        self.force_coef = force_coef
        self.torque_coef = torque_coef

        self.beta_rot = beta_rot
        
        self.downsample_traj = downsample_traj
    
    def transfer(self, regs, demos, plotting=False, plotting_std_dev=1.):
        if not self.sim:
            plotting = False
        handles = []
        if plotting:
            test_cloud = regs[0].test_scene_state.cloud
            test_color = regs[0].test_scene_state.color
            handles.append(self.sim.env.plot3(test_cloud[:,:3], 2, test_color if test_color is not None else (0,0,1)))
            self.sim.viewer.Step()
        
        aug_trajs = [demo.aug_traj for demo in demos]
        
        trajs = None
        active_lr = 'lr'
        for lr in active_lr:
            lr_trajs = transform_aug_trajs(lr, regs, aug_trajs)
            flip_angle_axis_in_place([traj[:,3:6] for traj in lr_trajs])
            if trajs is None:
                trajs = lr_trajs
            else:
                trajs = [np.c_[traj, lr_traj] for traj, lr_traj in zip(trajs, lr_trajs)]
        n_dof = trajs[0].shape[1]
        
        if plotting:
            for traj in trajs:
                color = np.r_[np.random.random(3),1]
                for i_lr, lr in enumerate(active_lr):
                    i_offset = i_lr * n_dof//len(active_lr)
                    handles.append(self.sim.env.drawlinestrip(traj[:,i_offset:i_offset+3], 2, color))
            self.sim.viewer.Step()
        
        if plotting:
            plt.ion()
            fig = plt.figure()
            for i_lr in range(len(active_lr)):
                i_offset = i_lr * n_dof//len(active_lr)
                for i_coord in range(n_dof//len(active_lr)):
                    plt.subplot(n_dof//len(active_lr), len(active_lr), i_coord*len(active_lr)+i_lr+1)
                    for traj in trajs:
                        plt.plot(traj[:, i_offset+i_coord])
            plt.draw()
        
        sys.stdout.write("aligning trajectories... ")
        sys.stdout.flush()
        coefs = np.r_[self.pos_coef, self.rot_coef, self.pos_vel_coef, self.rot_vel_coef, self.force_coef, self.torque_coef]
        if self.downsample_traj > 1:
            ds_trajs = [traj[::self.downsample_traj] for traj in trajs]
        else:
            ds_trajs = trajs
        if len(active_lr) == 1:
            dof_cost = lambda dof_val1, dof_val2: dof_val_cost(dof_val1, dof_val2, coefs)
        elif len(active_lr) == 2:
            h_n_dof = n_dof/2
            dof_cost = lambda dof_val1, dof_val2: dof_val_cost(dof_val1[:h_n_dof], dof_val2[:h_n_dof], coefs) + dof_val_cost(dof_val1[h_n_dof:], dof_val2[h_n_dof:], coefs)
        else:
            raise RuntimeError
        ds_trajs_timesteps_rs = align_trajs(ds_trajs, dof_cost)
        print "done"
        
        aligned_aug_trajs = []
        trajs_timesteps_rs = []
        for aug_traj, ds_traj_timesteps_rs in zip(aug_trajs, ds_trajs_timesteps_rs):
            traj_timesteps_rs0 = np.arange(self.downsample_traj*(len(ds_traj_timesteps_rs)-1)+1)/self.downsample_traj
            ds_traj_timesteps_rs, ds_traj_timesteps_rs0 = np.unique(ds_traj_timesteps_rs, return_index=True)
            ds_traj_timesteps_rs0 = ds_traj_timesteps_rs0.astype(float)
            ds_traj_timesteps_rs0[:-1] += (np.diff(ds_traj_timesteps_rs0)-1)/2
            traj_timesteps_rs = np.interp(traj_timesteps_rs0, ds_traj_timesteps_rs0, ds_traj_timesteps_rs*self.downsample_traj)
            aligned_aug_trajs.append(aug_traj.get_resampled_traj(traj_timesteps_rs))
            trajs_timesteps_rs.append(traj_timesteps_rs)
        trajs_timesteps_rs = np.asarray(trajs_timesteps_rs)
        
#         if plotting:
#             for i, (ds_traj_timesteps_rs, traj_timesteps_rs) in enumerate(zip(ds_trajs_timesteps_rs, trajs_timesteps_rs)):
#                 fig = plt.figure()
#                 plt.plot(np.arange(len(ds_traj_timesteps_rs))*self.downsample_traj, ds_traj_timesteps_rs*self.downsample_traj)
#                 plt.plot(traj_timesteps_rs)
        
        aligned_trajs = None
        for lr in active_lr:
            lr_aligned_trajs = transform_aug_trajs(lr, regs, aligned_aug_trajs)
            flip_angle_axis_in_place([traj[:,3:6] for traj in lr_aligned_trajs])
            if aligned_trajs is None:
                aligned_trajs = lr_aligned_trajs
            else:
                aligned_trajs = [np.c_[traj, lr_traj] for traj, lr_traj in zip(aligned_trajs, lr_aligned_trajs)]
        aligned_trajs = np.asarray(aligned_trajs)
        
        t_steps = aligned_trajs.shape[1]
        
        if plotting:
            fig = plt.figure()
            for i_lr in range(len(active_lr)):
                i_offset = i_lr * n_dof//len(active_lr)
                for i_coord in range(n_dof//len(active_lr)):
                    plt.subplot(n_dof//len(active_lr), len(active_lr), i_coord*len(active_lr)+i_lr+1)
                    for traj in aligned_trajs:
                        plt.plot(traj[:, i_offset+i_coord])
            plt.draw()
        
        dof_val_mus = np.empty((t_steps, n_dof))
        dof_val_sigmas = np.empty((t_steps, n_dof, n_dof))
        for t in range(t_steps):
            dof_val = aligned_trajs[:,t,:]
            dof_val_mus[t,:] = dof_val.mean(axis=0)
            dof_val_sigmas[t,:,:] = (dof_val-dof_val_mus[t,:]).T.dot(dof_val-dof_val_mus[t,:])/dof_val.shape[0]
        
        if plotting:
            for i_lr, lr in enumerate(active_lr):
                i_offset = i_lr * n_dof//len(active_lr)
                for t in range(t_steps)[::self.downsample_traj]:
                    pos = aligned_trajs[:,t,:3]
                    pos_mu = dof_val_mus[t,i_offset:i_offset+3]
                    pos_sigma = dof_val_sigmas[t,i_offset:i_offset+3,i_offset:i_offset+3]
                    U, s, V = np.linalg.svd(pos_sigma)
                    T = np.eye(4)
                    T[:3,:3] = plotting_std_dev * U * np.sqrt(s)
                    T[:3,3] = pos_mu
                    handles.append(self.sim.viewer.PlotEllipsoid(T, (0,1,0,1), True))
                    
                    rot_mu = dof_val_mus[t,i_offset+3:i_offset+6]
                    rot_sigma = dof_val_sigmas[t,i_offset+3:i_offset+6,i_offset+3:i_offset+6]
                    U, s, V = np.linalg.svd(rot_sigma)
                    rt_sigma_rot = (U * np.sqrt(s)).dot(U.T)
                    rot_sigma_pts = [rot_mu]
                    for i in range(3):
                        rot_sigma_pts.append(rot_mu + plotting_std_dev * rt_sigma_rot[:,i])
                        rot_sigma_pts.append(rot_mu - plotting_std_dev * rt_sigma_rot[:,i])
                    for sigma_pt in rot_sigma_pts:
                        hmat = np.eye(4)
                        hmat[:3,:3] = openravepy.rotationMatrixFromAxisAngle(sigma_pt)
                        hmat[:3,3] = pos_mu
                        handles.extend(sim_util.draw_axis(self.sim, hmat, arrow_length=.01, arrow_width=.001))
            self.sim.viewer.Step()
        
        ref_aug_traj = aug_traj.get_resampled_traj(np.arange(aligned_aug_trajs[0].n_steps)[::self.downsample_traj])
        
        manip_name = ""
        ee_link_names = []
        transformed_ee_trajs_rs = []
        init_traj = np.zeros((ref_aug_traj.n_steps,0))
        for lr in active_lr:
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            ee_link_name = "%s_gripper_tool_frame"%lr
            
            if manip_name:
                manip_name += "+"
            manip_name += arm_name
            ee_link_names.append(ee_link_name)
            
            init_traj = np.c_[init_traj, ref_aug_traj.lr2arm_traj[lr]] # initialize trajectory with that of the best demo
        
        new_ee_trajs = []
        for i_lr, lr in enumerate(active_lr):
            i_offset = i_lr * n_dof//len(active_lr)
            new_ee_traj = np.empty((t_steps,4,4))
            new_ee_traj[:] = np.eye(4)
            new_ee_traj[:,:3,3] = dof_val_mus[:,i_offset:i_offset+3]
            for t in range(t_steps):
                new_ee_traj[t,:3,:3] = openravepy.rotationMatrixFromAxisAngle(dof_val_mus[t,i_offset+3:i_offset+6])
            new_ee_trajs.append(new_ee_traj)
    
        new_ee_trajs = [new_ee_traj[::self.downsample_traj] for new_ee_traj in new_ee_trajs]
        
        print "planning pose trajectory following"
        test_traj, obj_value, pose_errs = planning.plan_follow_trajs(self.sim.robot, manip_name, ee_link_names, new_ee_trajs, init_traj, 
                                                                     start_fixed=False,
                                                                     use_collision_cost=self.use_collision_cost,
                                                                     beta_pos=self.beta_pos, beta_rot=self.beta_rot)
        
        # the finger trajectory is the same for the demo and the test trajectory
        for lr in active_lr:
            finger_name = "%s_gripper_l_finger_joint"%lr
            manip_name += "+" + finger_name
            test_traj = np.c_[test_traj, ref_aug_traj.lr2finger_traj[lr]]

        full_traj = (test_traj, sim_util.dof_inds_from_name(self.sim.robot, manip_name))
        test_aug_traj = demonstration.AugmentedTrajectory.create_from_full_traj(self.sim.robot, full_traj, lr2open_finger_traj=ref_aug_traj.lr2open_finger_traj, lr2close_finger_traj=ref_aug_traj.lr2close_finger_traj)
        
        if self.downsample_traj > 1:
            test_aug_traj = test_aug_traj.get_resampled_traj(np.arange(test_aug_traj.n_steps*self.downsample_traj)/self.downsample_traj)
        
        test_aug_traj.lr2dof_mu_traj = {}
        test_aug_traj.lr2dof_sigma_traj = {}
        test_aug_traj.lr2dof_trajs = {}
        for i_lr, lr in enumerate(active_lr):
            n_dof_lr = n_dof//len(active_lr)
            i_offset = i_lr * n_dof_lr
            test_aug_traj.lr2dof_mu_traj[lr] = dof_val_mus[:,i_offset:i_offset+n_dof_lr]
            test_aug_traj.lr2dof_sigma_traj[lr] = dof_val_sigmas[:,i_offset:i_offset+n_dof_lr,i_offset:i_offset+n_dof_lr]
            test_aug_traj.lr2dof_trajs[lr] = aligned_trajs[:,:,i_offset:i_offset+n_dof_lr]
        
        if plotting:
            for lr in active_lr:
                handles.append(self.sim.env.drawlinestrip(test_aug_traj.lr2ee_traj[lr][:,:3,3], 2, (0,0,1)))
            self.sim.viewer.Step()
        
        return test_aug_traj

def register_scenes(reg_factory, scene_state):
    sys.stdout.write("registering all scenes... ")
    sys.stdout.flush()
    regs = []
    demos = []
    for action, demo in reg_factory.demos.iteritems():
        reg = reg_factory.register(demo, scene_state)
        regs.append(reg)
        demos.append(demo)
    q_values, regs, demos = zip(*sorted([(reg.f._bending_cost, reg, demo) for (reg, demo) in zip(regs, demos)]))
    print "done"
    
    return regs, demos

def mass_calculate2(lr, robot, aligned_joint_traj):
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    arm = robot.GetManipulator(manip_name)
    
    joint_feedforward = [3.33, 1.16, 0.1, 0.25, 0.133, 0.0727, 0.0727]
    M_joint_inv = np.linalg.inv(np.diag(joint_feedforward))
    M_inv = []

    cur_traj = aligned_joint_traj
    M_inv = []
    for j in range(cur_traj.shape[0]):
        vals = cur_traj[j,:]
        robot.SetDOFValues(vals, arm.GetArmIndices())
        J = np.vstack((arm.CalculateJacobian(), arm.CalculateAngularVelocityJacobian()))
        M_inv.append(J.dot(M_joint_inv).dot(np.transpose(J)))
    M_inv = np.asarray(M_inv)
    return M_inv

def calculate_masses(test_aug_traj):
    M_inv = {}
    M_inv['l'] = []
    M_inv['r'] = []
    from rapprentice import PR2
    import rospy
    rospy.init_node("exec_task",disable_signals=True)
    pr2 = PR2.PR2()
    env = pr2.env
    robot = pr2.robot

    for lr in 'lr':
        M_inv[lr] = mass_calculate2(lr, robot, test_aug_traj.lr2arm_traj[lr])
    return M_inv

def calculate_costs(test_aug_traj):
    M_inv = calculate_masses(test_aug_traj)
    
    lr2Cts = {}
    lr2cts = {}
    for lr in 'lr':
        t_steps, n_dof = test_aug_traj.lr2dof_mu_traj[lr].shape
#         t_steps = test_aug_traj.n_steps #TODO
        assert n_dof == 18
        
        Mts = np.zeros((t_steps,18,18))
        mts = np.zeros((t_steps,18,1))
        Sts = np.zeros((t_steps,12,12))
        sts = np.zeros((t_steps,12,1))
        Fts = np.zeros((t_steps,12,12)) 
        fts = np.zeros((t_steps,12,1))
        Bts = np.zeros((t_steps,18,18))
        bts = np.zeros((t_steps,18,1))
        Nts = np.zeros((t_steps,12,12))
        Cts = np.zeros((t_steps,18,18))
        cts = np.zeros((t_steps,18,1))
        Lts = np.zeros((t_steps,18,18))
        Dts = np.zeros((t_steps,12,18))
        dt = 0.01
        Dts[:,:12,:12] = np.eye(12)
        Dts[:,:6,6:12] = dt
        Dts[:,:6,12:18] = (dt**2)*M_inv[lr][:t_steps] #TODO
        Dts[:,6:12,12:18] = dt*M_inv[lr][:t_steps]
    
        convergence = False
        isUp = True
        t = 0
        iter = 0
        while not convergence:
            if t % 100 == 0:
                print t
            
            empmu = np.zeros((18,1))
            empsigma = test_aug_traj.lr2dof_sigma_traj[lr][t]
            paddedCovs = test_aug_traj.lr2dof_sigma_traj[lr][t] + np.eye(18)*0.000001
            #clip eigen values here
            #Cts[t] = Mts[t] - Bts[t] #fix this clip eigen vluae
            pM = np.linalg.inv((paddedCovs + np.transpose(paddedCovs))/2)
            pm = -pM.dot(empmu)
            
            FtPadded = np.zeros((18,18))
            FtPadded[0:12,0:12] = Fts[t]
            pD = FtPadded + Bts[t]
    
            ftpadded = np.zeros((18,1))
            ftpadded[0:12] = fts[t]
            pd = ftpadded + bts[t]
            
            Cts[t], cts[t], Lts[t] = processFuncCt("LBFGS", pM, pm, empsigma, empmu, pD, pd, Cts[t], cts[t], Lts[t])
            #cts[t] = processFuncct(bts[t], fts[t], "default") #TODO make this cov inv times mean(always 0)
            
            if t == 0:
                Mts[t] = np.linalg.inv((paddedCovs + np.transpose(paddedCovs))/2) # true always
                mts[t] = np.zeros((18,1))
            else:
                Mts[t] = FtPadded + Cts[t] + Bts[t]
                mts[t] = ftpadded + cts[t] + bts[t]
                #Use Ct to get Mt and mt
                Mt = Mts[t]
                Mxx = Mt[0:12, 0:12]
                Muu = Mt[12:18, 12:18]
                Mxu = Mt[0:12, 12:18]
                Mux = Mt[12:18, 0:12]
                Sts[t] = Mxx - Mxu.dot(np.linalg.inv(Muu)).dot(Mux)
                sts[t] = mts[t][0:12] - Mxu.dot(np.linalg.inv(Muu)).dot(mts[t][12:18])
                #Use Mt, mt to get St, st
                Bts[t-1] = np.transpose(Dts[t-1]).dot(Sts[t] - Fts[t]).dot(Dts[t-1])
                bts[t-1] = np.transpose(Dts[t-1]).dot(sts[t] - fts[t])
    
            if t < t_steps-1:
                Fts[t+1] = np.linalg.inv(Dts[t].dot(np.linalg.inv(Mts[t] - Bts[t])).dot(np.transpose(Dts[t])) + Nts[t]) #TODO pick that manually
                fts[t+1] = Fts[t+1].dot(Dts[t]).dot(np.linalg.inv(Mts[t] - Bts[t])).dot(mts[t] - bts[t])
            
            if t == t_steps-1:
                isUp = False
            if t == 0:
                isUp = True
                iter += 1
            if isUp:
                t += 1
            else:
                t -= 1
            
            #TODO norm of difference less than a value, fraction of differences, convergence on C's/M's
            if iter == 4:
                break
        
        lr2Cts[lr] = Cts
        lr2cts[lr] = cts
    return lr2Cts, lr2cts

def processFuncCt(cliptype, pM, pm, empsig, empmu, pD, pd, Ct, ct, Lt):
    if cliptype == "LBFGS":
        def func_minimize(p, empsig, pD):
            L = np.zeros(pD.shape)
            tidx = np.triu_indices(pD.shape[0])
            L[tidx] = p
            Ct = L.T.dot(L)

            prior_m_x = 1e-1
            prior_m_u = 1e-2
             
            prior_wt = 1
            prior_mat = np.diag(np.r_[prior_m_x*np.ones(12), prior_m_u*np.ones(6)])
             
            val = ((empsig + prior_wt*prior_mat)*Ct).sum() \
                - prior_wt*np.log(max(1e-100,np.linalg.det(Ct))) \
                - np.log(max(1e-100,np.linalg.det(Ct+pD)))
            grad = 2*L.dot(-empsig - prior_wt*prior_mat + prior_wt*np.linalg.inv(Ct) + np.linalg.inv(Ct + pD))
            grad = -grad[tidx]
            return val, grad
        
        if not np.any(Ct):
            Ct = pM - pD
        ct = pm - pd;
        
        if not np.any(Lt):
            nval = 1e-4
            mval = 1e3
            
            val, vec = np.linalg.eig((Ct+Ct.T)/2.)
            val = np.real(val)
            vec = np.real(vec)
            val = np.clip(val, nval, mval)
            val = np.diag(val)
            Ct = vec.dot(val).dot(vec.T)
            Lt = scipy.linalg.cholesky((Ct+Ct.T)/2.)
        
        tidx = np.triu_indices(Ct.shape[0])
        theta, _, _ = optimize.fmin_l_bfgs_b(func_minimize, Lt[tidx], fprime=None, args=(empsig, pD), maxfun=20, maxiter=20)
        
        Lt[tidx] = theta
        Ct = Lt.T.dot(Lt)

        ctm = np.linalg.solve(Lt, (np.linalg.solve(Lt.T,((Ct + pD).dot(empmu + np.linalg.solve(Ct + pD, pd))))))
        ct = -Ct.dot(ctm)

        return Ct, ct, Lt
    else:
        raise NotImplementedError
    
class ForceAugmentedTrajectory(AugmentedTrajectory):
    def __init__(self, lr2force_traj, lr2torque_traj, lr2arm_traj=None, lr2finger_traj=None, lr2ee_traj=None, lr2open_finger_traj=None, lr2close_finger_traj=None):
        super(ForceAugmentedTrajectory, self).__init__(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, 
                                                       lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        for lr2traj in [lr2force_traj, lr2torque_traj]:
            if lr2traj is None:
                continue
            for lr in lr2traj.keys():
                assert lr2traj[lr].shape[0] == self.n_steps
        
        self.lr2force_traj = lr2force_traj
        self.lr2torque_traj = lr2torque_traj
    
    def get_resampled_traj(self, timesteps_rs):
        """
        The t step of the resampled trajectory corresponds to the timesteps_rs[t] step of aug_traj. If timesteps_rs[t] is fractional, the appropiate interpolation is used.
        The domain of each element in timesteps_rs is between 0 and aug_traj.n_steps-1 and the length of the resampled trajectory is len(timesteps_rs).
        """
        aug_traj = super(ForceAugmentedTrajectory, self).get_resampled_traj(timesteps_rs)
        lr2force_traj_rs = None if self.lr2force_traj is None else {}
        lr2torque_traj_rs = None if self.lr2torque_traj is None else {}
        for (lr2traj_rs, self_lr2traj) in [(lr2force_traj_rs, self.lr2force_traj), (lr2torque_traj_rs, self.lr2torque_traj)]:
            if self_lr2traj is None:
                continue
            for lr in self_lr2traj.keys():
                lr2traj_rs[lr] = math_utils.interp2d(timesteps_rs, np.arange(len(self_lr2traj[lr])), self_lr2traj[lr])
        return ForceAugmentedTrajectory(lr2force_traj_rs, lr2torque_traj_rs, lr2arm_traj=aug_traj.lr2arm_traj, lr2finger_traj=aug_traj.lr2finger_traj, lr2ee_traj=aug_traj.lr2ee_traj, 
                                 lr2open_finger_traj=aug_traj.lr2open_finger_traj, lr2close_finger_traj=aug_traj.lr2close_finger_traj)

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='../.win_prop.txt')
    parser.add_argument("--std_dev", type=float, default=1, help="number of standard deviations plotted for the covariance")

    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str)
    parser_eval.add_argument("reg_type", type=str, choices=['segment', 'rpm', 'bij'], default='bij')
    parser_eval.add_argument("--downsample_size", type=float, default=0.025)
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser_eval.add_argument("--gpu", action="store_true", default=False)

    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=100.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--use_collision_cost", type=int, default=1)

    parser_eval.add_argument("--pos_coef", type=float, default=1, help="coefficient for dtw position cost")
    parser_eval.add_argument("--rot_coef", type=float, default=.1, help="coefficient for dtw rotation cost")
    parser_eval.add_argument("--pos_vel_coef", type=float, default=0, help="coefficient for dtw position velocity cost")
    parser_eval.add_argument("--rot_vel_coef", type=float, default=0, help="coefficient for dtw rotation velocity cost")
    parser_eval.add_argument("--force_coef", type=float, default=1, help="coefficient for dtw force cost")
    parser_eval.add_argument("--torque_coef", type=float, default=1, help="coefficient for dtw torque cost")

    parser_eval.add_argument("--downsample_traj", type=int, default=1, help="downsample demonstration trajectory by this factor")
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
            if "%s_gripper_force"%lr in seg_info:
                lr2force_traj[lr] = np.asarray(seg_info["%s_gripper_force"%lr][:,:3])
                lr2torque_traj[lr] = np.asarray(seg_info["%s_gripper_force"%lr][:,3:])
        aug_traj = ForceAugmentedTrajectory(lr2force_traj, lr2torque_traj, lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
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
    
    # for now, use the cloud of the first demo as the current cloud
    full_cloud = demos.values()[0].scene_state.cloud
    scene_state = demonstration.SceneState(full_cloud, downsample_size=args.eval.downsample_size)

    regs, demos = register_scenes(reg_factory, scene_state)
    
    trajectory_transferer = MultipleDemosPoseTrajectoryTransferer(sim, args.eval.pos_coef, args.eval.rot_coef, args.eval.pos_vel_coef, args.eval.rot_vel_coef, args.eval.force_coef, args.eval.torque_coef, 
                                                                  args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost, 
                                                                  downsample_traj=args.eval.downsample_traj)
    n_demos = min(args.eval.max_num_demos, len(reg_factory.demos))
    test_aug_traj = trajectory_transferer.transfer(regs[:n_demos], demos[:n_demos], plotting=args.animation, plotting_std_dev=args.std_dev)

    lr2Cts, lr2cts = calculate_costs(test_aug_traj)
    
    lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)

if __name__ == "__main__":
    main()
