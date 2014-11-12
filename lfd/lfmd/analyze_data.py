#!/usr/bin/env python

from __future__ import division

import sys
import os.path
import h5py
import copy
import openravepy

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from pykalman import KalmanFilter
from pykalman.standard import _last_dims
import trajoptpy
import IPython as ipy

from lfd.demonstration import demonstration
from lfd.rapprentice import util, math_utils, planning, resampling
from lfd.environment import sim_util
from lfd.demonstration.demonstration import SceneState, AugmentedTrajectory, Demonstration
from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject
from lfd.environment.environment import LfdEnvironment
from lfd.registration.registration import TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory
from lfd.registration.registration_gpu import BatchGpuTpsRpmBijRegistrationFactory, BatchGpuTpsRpmRegistrationFactory
from lfd.transfer.transfer import TrajectoryTransferer


DT = 0.01

def dof_val_cost(aug_traj1, aug_traj2, t1, t2, lr, coefs):
    t1_v = t1+1 if t1 == 0 else t1
    t2_v = t2+1 if t2 == 0 else t2
    cost = 0.0
    if coefs[0] != 0:
        cost = coefs[0] * np.linalg.norm(aug_traj2.lr2ee_traj[lr][t2,:3,3] - aug_traj1.lr2ee_traj[lr][t1,:3,3])
    if coefs[1] != 0:
        rot1 = aug_traj1.lr2ee_traj[lr][t1,:3,:3]
        rot2 = aug_traj2.lr2ee_traj[lr][t2,:3,:3]
        aa_diff = openravepy.axisAngleFromRotationMatrix(rot2.dot(rot1.T))
        cost += coefs[1] * np.linalg.norm(aa_diff)
    if coefs[2] != 0:
        vel1 = (aug_traj1.lr2ee_traj[lr][t1_v,:3,3] - aug_traj1.lr2ee_traj[lr][t1_v-1,:3,3]) / DT
        vel2 = (aug_traj2.lr2ee_traj[lr][t2_v,:3,3] - aug_traj2.lr2ee_traj[lr][t2_v-1,:3,3]) / DT
        cost = coefs[2] * np.linalg.norm(vel2 - vel1)
    if coefs[3] != 0:
        t1_v = t1+1 if t1 == 0 else t1
        t2_v = t2+1 if t2 == 0 else t2
        rot_vel1 = (aug_traj1.lr2ee_traj[lr][t1_v,:3,:3].dot(aug_traj1.lr2ee_traj[lr][t1_v-1,:3,:3].T)) / DT
        rot_vel2 = (aug_traj2.lr2ee_traj[lr][t2_v,:3,:3].dot(aug_traj2.lr2ee_traj[lr][t2_v-1,:3,:3].T)) / DT
        aa_diff = openravepy.axisAngleFromRotationMatrix(rot_vel2.dot(rot_vel1.T))
        cost += coefs[3] * np.linalg.norm(aa_diff)
    if aug_traj1.lr2force_traj and aug_traj2.lr2force_traj:
        if coefs[4] != 0:
            cost += coefs[4] * np.linalg.norm(aug_traj2.lr2force_traj[lr][t2,:3] - aug_traj1.lr2force_traj[lr][t1,:3])
        if coefs[5] != 0:
            cost += coefs[5] * np.linalg.norm(aug_traj2.lr2force_traj[lr][t2,3:] - aug_traj1.lr2force_traj[lr][t1,3:])
    return cost

def dtw(aug_traj1, aug_traj2, dof_cost):
    n = aug_traj1.n_steps
    m = aug_traj2.n_steps
    DTW = np.zeros((n+1, m+1))
    DTW[:,0] = np.inf
    DTW[0,:] = np.inf
    DTW[0, 0] = 0
    pointers = np.empty((n+1, m+1), dtype=object)
    pointers.fill(None)
    t1_pen = np.zeros((n+1, m+1))
    t2_pen = np.zeros((n+1, m+1))
    for t1 in range(n):
        #if t1 % 100 == 0: print t1
        t1 = t1+1 # increase by one because we need to 1 index DTW
        for t2 in range(m):
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
            DTW[t1, t2] = dof_cost(aug_traj1, aug_traj2, t1-1, t2-1) + best_next# + t1_pen[t1, t2] + t2_pen[t1, t2]
    return DTW, pointers, t1_pen, t2_pen

def align_trajs(aug_trajs, dof_cost):
    trajs_timesteps_rs = []
    for i_demo, aug_traj in enumerate(aug_trajs):
        if i_demo == 0:
            trajs_timesteps_rs.append(range(aug_traj.n_steps))
        else:
            t1_pens = []
            t2_pens = []
            DTW, pointers, t1_pen, t2_pen = dtw(aug_trajs[0], aug_traj, dof_cost)
            traj_timesteps_rs = [[] for _ in range(aug_trajs[0].n_steps)]
            next_ij = (aug_trajs[0].n_steps, aug_traj.n_steps)
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
            angle_traj[0] = -1 * (angle_traj[0] % (2*np.pi)) + 2*np.pi
    # flip axes for the rest of the trajectory for each trajectory
    for axis_traj, angle_traj in zip(axis_trajs, angle_trajs):
        for t in range(len(axis_traj)):
            if t > 0 and axis_traj[t].dot(axis_traj[t-1]) < 0:
                axis_traj[t] *= -1
                angle_traj[t] %= 2*np.pi
                angle_traj[t] *= -1
                angle_traj[t] += 2*np.pi
    # TODO: check if this is equivalent
#     for axis_traj, angle_traj in zip(axis_trajs, angle_trajs):
#         dot_products = np.einsum('ij,ij->t', axis_traj[:-1], axis_traj[1:]) # pairwise dot products axis_traj[t].dot(axis_traj[t-1])
#         flip_inds = np.r_[False, dot_products < 0]
#         for flip_ind in  np.where(flip_inds)[0]:
#             if flip_ind != len(flip_inds)-1:
#                 flip_inds[flip_ind+1:] = np.invert(flip_inds[flip_ind+1:])
#         axis_traj[flip_inds] *= -1
#         angle_traj[flip_inds] = -1 * (angle_traj[flip_inds] % 2*np.pi) + 2*np.pi
    # make all angles be between 0 and 2*np.pi
    for angle_traj in angle_trajs:
        angle_traj[:] = np.unwrap(angle_traj)
    return angle_trajs, axis_trajs

def flip_angle_axis_in_place(aa_trajs):
    orig_rot_trajs = [rotationMatrixFromAxisAngle(aa_traj) for aa_traj in aa_trajs]

    angle_trajs, axis_trajs = flipped_angle_axis(aa_trajs)
    for aa_traj, axis_traj, angle_traj in zip(aa_trajs, axis_trajs, angle_trajs):
        aa_traj[:,:] = axis_traj * angle_traj[:,None]
    
    # check rotations are the same
    new_rot_trajs = [rotationMatrixFromAxisAngle(aa_traj) for aa_traj in aa_trajs]
    for orig_rot_traj, new_rot_traj in zip(orig_rot_trajs, new_rot_trajs):
#         rot_diffs = [orig_rot.T.dot(new_rot) for orig_rot, new_rot in zip(orig_rot_traj, new_rot_traj)]
#         aa_diffs = axisAngleFromRotationMatrix(rot_diffs)
#         angle_diffs = np.apply_along_axis(np.linalg.norm, 1, aa_diffs)
        assert np.allclose(orig_rot_traj, new_rot_traj)
    return aa_trajs

def axisAngleFromRotationMatrix(rots):
    aa_diffs = np.empty((len(rots), 3))
    for t, rot in enumerate(rots):
        aa_diffs[t,:] = openravepy.axisAngleFromRotationMatrix(rot)
    return aa_diffs

def rotationMatrixFromAxisAngle(aas):
    rots = [openravepy.rotationMatrixFromAxisAngle(aa) for aa in aas]
    rots = np.asarray(rots)
    return rots

def diff_rotation_matrix(rots):
    rot_diffs = np.empty((len(rots)-1, 3, 3))
    for t in range(len(rots)-1):
        rot_diffs[t,:,:] = rots[t+1,:3,:3].dot(rots[t,:3,:3].T)
    return rot_diffs

def stack_traj(aug_traj):
    """
    Stacks trajectory in the following order: position, velocity, force and external force
    """
    lr2traj = {}
    for lr in aug_traj.lr2ee_traj.keys():
        pos_traj = np.c_[aug_traj.lr2ee_traj[lr][:,:3,3], axisAngleFromRotationMatrix(aug_traj.lr2ee_traj[lr][:,:3,:3])]
        vel_traj = np.c_[np.diff(aug_traj.lr2ee_traj[lr][:,:3,3], axis=0),
                         axisAngleFromRotationMatrix(diff_rotation_matrix(aug_traj.lr2ee_traj[lr][:,:3,:3]))]
        vel_traj = np.r_[vel_traj[0][None,:], vel_traj]
        vel_traj /= DT
        force_traj = np.zeros((aug_traj.n_steps,0))
        if aug_traj.lr2force_traj:
            force_traj = aug_traj.lr2force_traj[lr]
        lr2traj[lr] = np.c_[pos_traj, vel_traj, force_traj]
    return lr2traj

def flip_wrist_rotations(aug_trajs):
    T_x = np.eye(4) # transformation that rotates by pi around the x axis
    T_x[:3,:3] = openravepy.rotationMatrixFromAxisAngle(np.r_[1, 0, 0] * np.pi)
    for lr in aug_trajs[0].lr2ee_traj.keys():
        hmat_0l = aug_trajs[0].lr2ee_traj[lr][-1]
        for aug_traj in aug_trajs[1:]:
            hmat_il = aug_traj.lr2ee_traj[lr][-1]
            angle_diff = np.linalg.norm(openravepy.axisAngleFromRotationMatrix(hmat_il[:3,:3].dot(hmat_0l[:3,:3].T)))
            angle_diff_flipped = np.linalg.norm(openravepy.axisAngleFromRotationMatrix((hmat_il.dot(T_x)[:3,:3]).dot(hmat_0l[:3,:3].T)))
            if angle_diff_flipped < angle_diff:
                aug_traj.lr2ee_traj[lr] = np.asarray([hmat.dot(T_x) for hmat in aug_traj.lr2ee_traj[lr]]) # rotate ee_traj by T_x

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
        
        trajs = None
        active_lr = 'lr'

        if plotting:
            for demo in demos:
                aug_traj = demo.aug_traj
                color = np.r_[np.random.random(3),1]
                for lr in active_lr:
                    handles.append(self.sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 2, color))
            self.sim.viewer.Idle()
        
        aug_trajs = [demo.aug_traj.get_transformed_traj(reg) for (reg, demo) in zip(regs, demos)]
#         for lr in active_lr:
#             for aug_traj in aug_trajs:
#                 for t in range(aug_traj.n_steps)[::self.downsample_traj]:
#                     handles.extend(sim_util.draw_axis(self.sim, aug_traj.lr2ee_traj[lr][t], arrow_length=.01, arrow_width=.001))
        
        if plotting:
            for aug_traj in aug_trajs:
                color = np.r_[np.random.random(3),1]
                for lr in active_lr:
                    handles.append(self.sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 2, color))
            self.sim.viewer.Step()
            
            lr2trajs = {}
            for aug_traj in aug_trajs:
                lr2traj = stack_traj(aug_traj)
                flip_angle_axis_in_place([traj[::-1,3:6] for traj in lr2traj.values()]) # pass in reverse trajectory in order to align with respact to last time step
                for lr in active_lr:
                    if lr not in lr2trajs:
                        lr2trajs[lr] = []
                    lr2trajs[lr].append(lr2traj[lr])
            n_dof = lr2trajs.values()[0][0].shape[1] # n_dof is per arm
        
            plt.ion()
            fig = plt.figure()
            for i_lr, lr in enumerate(active_lr):
                for i_dof in range(n_dof):
                    plt.subplot(n_dof, len(active_lr), i_dof*len(active_lr)+i_lr+1)
                    for traj in lr2trajs[lr]:
                        plt.plot(traj[:,i_dof])
            plt.draw()
        
        sys.stdout.write("aligning trajectories... ")
        sys.stdout.flush()
        coefs = np.r_[self.pos_coef, self.rot_coef, self.pos_vel_coef, self.rot_vel_coef, self.force_coef, self.torque_coef]
        if self.downsample_traj > 1:
            ds_aug_trajs = [aug_traj.get_resampled_traj(np.arange(aug_traj.n_steps)[::self.downsample_traj]) for aug_traj in aug_trajs]
        else:
            ds_aug_trajs = aug_trajs
        if len(active_lr) == 1:
            dof_cost = lambda aug_traj1, aug_traj2, t1, t2: dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[0], coefs)
        elif len(active_lr) == 2:
            dof_cost = lambda aug_traj1, aug_traj2, t1, t2: dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[0], coefs) + dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[1], coefs)
        else:
            raise RuntimeError
        ds_trajs_timesteps_rs = align_trajs(ds_aug_trajs, dof_cost)
        print "done"
        
        aligned_aug_trajs = []
        trajs_timesteps_rs = []
        for aug_traj, ds_traj_timesteps_rs in zip(aug_trajs, ds_trajs_timesteps_rs):
            traj_timesteps_rs0 = np.arange(self.downsample_traj*(len(ds_traj_timesteps_rs)-1)+1)
            ds_traj_timesteps_rs, ds_traj_timesteps_rs0 = np.unique(ds_traj_timesteps_rs, return_index=True)
            ds_traj_timesteps_rs0 = ds_traj_timesteps_rs0.astype(float)
            ds_traj_timesteps_rs0[:-1] += (np.diff(ds_traj_timesteps_rs0)-1)/2
            traj_timesteps_rs = np.interp(traj_timesteps_rs0, ds_traj_timesteps_rs0*self.downsample_traj, ds_traj_timesteps_rs*self.downsample_traj)
            aligned_aug_trajs.append(aug_traj.get_resampled_traj(traj_timesteps_rs))
            trajs_timesteps_rs.append(traj_timesteps_rs)
        trajs_timesteps_rs = np.asarray(trajs_timesteps_rs)
        
#         if plotting:
#             for i, (ds_traj_timesteps_rs, traj_timesteps_rs) in enumerate(zip(ds_trajs_timesteps_rs, trajs_timesteps_rs)):
#                 fig = plt.figure()
#                 plt.plot(np.arange(len(ds_traj_timesteps_rs))*self.downsample_traj, ds_traj_timesteps_rs*self.downsample_traj)
#                 plt.plot(traj_timesteps_rs)
        
        aligned_lr2trajs = {}
        for aug_traj in aligned_aug_trajs:
            lr2traj = stack_traj(aug_traj)
            flip_angle_axis_in_place([traj[::-1,3:6] for traj in lr2traj.values()]) # pass in reverse trajectory in order to align with respact to last time step
            for lr in active_lr:
                if lr not in aligned_lr2trajs:
                    aligned_lr2trajs[lr] = []
                aligned_lr2trajs[lr].append(lr2traj[lr])
        for lr in active_lr:
            aligned_lr2trajs[lr] = np.asarray(aligned_lr2trajs[lr])
            t_steps = aligned_lr2trajs[lr].shape[1]
        
        if plotting:
            fig = plt.figure()
            for i_lr, lr in enumerate(active_lr):
                for i_dof in range(n_dof):
                    plt.subplot(n_dof, len(active_lr), i_dof*len(active_lr)+i_lr+1)
                    for traj in aligned_lr2trajs[lr]:
                        plt.plot(traj[:,i_dof])
            plt.draw()
        
#         for lr in active_lr:
#             for t in range(t_steps)[::self.downsample_traj]:
#                 for i_demo in range(dof_val.shape[0]):
#                     dof_val = aligned_lr2trajs[lr][:,t,:]
#                     hmat = np.eye(4)
#                     hmat[:3,:3] = openravepy.rotationMatrixFromAxisAngle(dof_val[i_demo,3:6])
#                     hmat[:3,3] = dof_val[i_demo,:3]
#                     handles.extend(sim_util.draw_axis(self.sim, hmat, arrow_length=.01, arrow_width=.001))

        lr2dof_mu_traj = {}
        lr2dof_sigma_traj = {}
        for lr in active_lr:
            n_dof = aligned_lr2trajs[lr].shape[2]
            dof_mu_traj = np.empty((t_steps, n_dof))
            dof_sigma_traj = np.empty((t_steps, n_dof, n_dof))
            for t in range(t_steps):
                dof_val = aligned_lr2trajs[lr][:,t,:]
                dof_mu_traj[t,:] = dof_val.mean(axis=0)
                dof_sigma_traj[t,:,:] = (dof_val-dof_mu_traj[t,:]).T.dot(dof_val-dof_mu_traj[t,:])/dof_val.shape[0]
                
                # recompute rotation means
                rot0 = openravepy.rotationMatrixFromAxisAngle(dof_val[0,3:6])
                aa_devs = [np.zeros(3)] # deviation from the rotation of demo 0
                for i_demo in range(1, dof_val.shape[0]):
                    rot = openravepy.rotationMatrixFromAxisAngle(dof_val[i_demo,3:6])
                    rot_dev = rot0.T.dot(rot)
                    aa_devs.append(openravepy.axisAngleFromRotationMatrix(rot_dev))
                aa_devs = np.asarray(aa_devs)
                rot_mu_dev = openravepy.rotationMatrixFromAxisAngle(aa_devs.mean(axis=0))
                rot_mu = rot0.dot(rot_mu_dev)
                dof_mu_traj[t,3:6] = openravepy.axisAngleFromRotationMatrix(rot_mu)
                # recompute rotation covariances
                aa_mu_devs = [] # deviation from the mean rotation
                for i_demo in range(dof_val.shape[0]):
                    rot = openravepy.rotationMatrixFromAxisAngle(dof_val[i_demo,3:6])
                    rot_mu_dev = rot_mu.T.dot(rot)
                    aa_mu_devs.append(openravepy.axisAngleFromRotationMatrix(rot_mu_dev))
                aa_mu_devs = np.asarray(aa_mu_devs)
                dof_sigma_traj[t,3:6,3:6] = aa_mu_devs.T.dot(aa_mu_devs)/dof_val.shape[0]
                
            lr2dof_mu_traj[lr] = dof_mu_traj
            lr2dof_sigma_traj[lr] = dof_sigma_traj

#         for lr in active_lr:
#             for t in range(t_steps)[::self.downsample_traj]:
#                 for i_demo in range(dof_val.shape[0]):
#                     dof_val = aligned_lr2trajs[lr][:,t,:]
#                     hmat = np.eye(4)
#                     hmat[:3,:3] = openravepy.rotationMatrixFromAxisAngle(dof_val[i_demo,3:6])
#                     hmat[:3,3] = dof_val[i_demo,:3]
#                     handles.extend(sim_util.draw_axis(self.sim, hmat, arrow_length=.01, arrow_width=.001))
        
        if plotting:
            for lr in active_lr:
                for t in range(t_steps)[::self.downsample_traj]:
                    pos_mu = lr2dof_mu_traj[lr][t,:3]
                    pos_sigma = lr2dof_sigma_traj[lr][t,:3,:3]
                    U, s, V = np.linalg.svd(pos_sigma)
                    T = np.eye(4)
                    T[:3,:3] = plotting_std_dev * U * np.sqrt(s)
                    T[:3,3] = pos_mu
                    handles.append(self.sim.viewer.PlotEllipsoid(T, (0,1,0,1), True))
                    
                    rot_mu = lr2dof_mu_traj[lr][t,3:6]
                    rot_sigma = lr2dof_sigma_traj[lr][t,3:6,3:6]
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
        
        manip_name = ""
        ee_link_names = []
        transformed_ee_trajs_rs = []
        init_traj = np.zeros((ds_aug_trajs[0].n_steps,0))
        for lr in active_lr:
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            ee_link_name = "%s_gripper_tool_frame"%lr
            
            if manip_name:
                manip_name += "+"
            manip_name += arm_name
            ee_link_names.append(ee_link_name)
            
            init_traj = np.c_[init_traj, ds_aug_trajs[0].lr2arm_traj[lr]] # initialize trajectory with that of the best demo
        
        new_ee_trajs = []
        for lr in active_lr:
            new_ee_traj = np.empty((t_steps,4,4))
            new_ee_traj[:] = np.eye(4)
            new_ee_traj[:,:3,3] = lr2dof_mu_traj[lr][:,:3]
            for t in range(t_steps):
                new_ee_traj[t,:3,:3] = openravepy.rotationMatrixFromAxisAngle(lr2dof_mu_traj[lr][t,3:6])
            new_ee_traj = np.asarray(resampling.interp_hmats(np.arange(ds_aug_trajs[0].n_steps)*self.downsample_traj, np.arange(len(new_ee_traj)), new_ee_traj))
            new_ee_trajs.append(new_ee_traj)
        
        print "planning pose trajectory following"
        test_traj, obj_value, pose_errs = planning.plan_follow_trajs(self.sim.robot, manip_name, ee_link_names, new_ee_trajs, init_traj, 
                                                                     start_fixed=False,
                                                                     use_collision_cost=self.use_collision_cost,
                                                                     beta_pos=self.beta_pos, beta_rot=self.beta_rot)
        
        # the finger trajectory is the same for the demo and the test trajectory
        for lr in active_lr:
            finger_name = "%s_gripper_l_finger_joint"%lr
            manip_name += "+" + finger_name
            test_traj = np.c_[test_traj, ds_aug_trajs[0].lr2finger_traj[lr]]

        full_traj = (test_traj, sim_util.dof_inds_from_name(self.sim.robot, manip_name))
        test_aug_traj = ForceAugmentedTrajectory.create_from_full_traj(self.sim.robot, full_traj, None, None)
        
        if self.downsample_traj > 1:
            test_aug_traj = test_aug_traj.get_resampled_traj(np.arange(self.downsample_traj*(test_aug_traj.n_steps-1)+1)/self.downsample_traj)
        
        test_aug_traj.lr2open_finger_traj = aligned_aug_trajs[0].lr2open_finger_traj
        test_aug_traj.lr2close_finger_traj = aligned_aug_trajs[0].lr2close_finger_traj
        test_aug_traj.lr2force_traj = {}
        test_aug_traj.lr2x_traj = {}
        for lr in active_lr:
            assert lr2dof_mu_traj[lr].shape[0] == test_aug_traj.n_steps
            assert lr2dof_mu_traj[lr].shape[1] == 18
            test_aug_traj.lr2force_traj[lr] = lr2dof_mu_traj[lr][:,12:18]
        
        test_aug_traj.lr2dof_mu_traj = lr2dof_mu_traj
        test_aug_traj.lr2dof_sigma_traj = lr2dof_sigma_traj
        test_aug_traj.lr2dof_trajs = aligned_lr2trajs
        
        if plotting:
            for lr in active_lr:
                handles.append(self.sim.env.drawlinestrip(test_aug_traj.lr2ee_traj[lr][:,:3,3], 2, (0,0,1)))
            self.sim.viewer.Step()
        
#         from rapprentice import eval_util
#         f = h5py.File("towel_0_0_processed.h5")
#         M_inv = {}
#         J = {}
#         for lr in active_lr:
#             M_inv[lr], J[lr] = mass_calculate3(lr, self.sim.robot, test_aug_traj.lr2arm_traj[lr])
#         eval_util.add_obj_to_group(f, 'M_inv', M_inv)
#         eval_util.add_obj_to_group(f, 'J', J)
#         eval_util.add_obj_to_group(f, 'dof_mu_traj', lr2dof_mu_traj)
#         eval_util.add_obj_to_group(f, 'dof_sigma_traj', lr2dof_sigma_traj)
#         eval_util.add_obj_to_group(f, 'aligned_trajs', aligned_lr2trajs)

        return test_aug_traj

def kalman_smoother(Z, M_inv, plotting=False):
    """
    X: state
    U: control
    Z: observation (position and forces)
    F: state transition model
    B: control input model
    Q: process variance
    R: observation variance
    """
    t_steps = Z.shape[0]
    
    x0 = np.r_[Z[0,0:6],
               np.zeros(4*6)]
    P0 = np.eye(5*6)
    P0[0*6:1*6,0*6:1*6] *= 0.005 # small
    P0[1*6:2*6,1*6:2*6] *= 0     # small
    P0[2*6:3*6,2*6:3*6] *= 0     # small
    P0[3*6:4*6,3*6:4*6] *= 100   # medium # TODO
    P0[4*6:5*6,4*6:5*6] *= 1     # high   # TODO
    
    # transition matrix
    F = np.zeros((t_steps-1, 5*6, 5*6))
    # observation matrix
    H = np.r_[np.c_[      np.eye(6), np.zeros((6,6)),   np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))],
              np.c_[np.zeros((6,6)), np.zeros((6,6)),   np.zeros((6,6)), np.zeros((6,6)),       np.eye(6)]]
    for t in range(t_steps-1):
        F[t] = np.r_[np.c_[      np.eye(6),    DT*np.eye(6), (DT**2)*np.eye(6), np.zeros((6,6)), np.zeros((6,6))],
                     np.c_[np.zeros((6,6)),       np.eye(6),      DT*np.eye(6), np.zeros((6,6)), np.zeros((6,6))],
                     np.c_[np.zeros((6,6)), np.zeros((6,6)),   np.zeros((6,6)),        M_inv[t],        M_inv[t]],
                     np.c_[np.zeros((6,6)), np.zeros((6,6)),   np.zeros((6,6)),       np.eye(6), np.zeros((6,6))],
                     np.c_[np.zeros((6,6)), np.zeros((6,6)),   np.zeros((6,6)), np.zeros((6,6)),       np.eye(6)]]
    
    # transition covariance
    Q = np.eye(5*6)
    Q[0*6:1*6,0*6:1*6] *= 0.005       # small
    Q[1*6:2*6,1*6:2*6] *= (0.005/DT)  # small
    Q[2*6:3*6,2*6:3*6] *= (1/(DT*DT)) # small
    Q[3*6:4*6,3*6:4*6] *= 100         # medium # TODO
    Q[4*6:5*6,4*6:5*6] *= 1           # high   # TODO
    # observation covariance
    R = np.eye(2*6)
    R[0*6:1*6,0*6:1*6] *= 0.005
    R[1*6:2*6,1*6:2*6] *= 5 # TODO
    
    def em_transition_matrix(transition_offsets, smoothed_state_means,
                              smoothed_state_covariances, pairwise_covariances):
        res = F.copy()
        n_timesteps, n_dim_state, _ = smoothed_state_covariances.shape
        print "em_transition_matrix"
        import time
        time_start = time.time()
        for tt in range(1, n_timesteps):
            if tt % 100 == 0:
                print tt
            t_start = np.clip(tt-500, 1, n_timesteps)
            t_end = np.clip(tt+500+1, 1, n_timesteps)
            
            res1 = np.zeros((n_dim_state, n_dim_state))
            res2 = np.zeros((n_dim_state, n_dim_state))
            ws = np.exp(-((np.arange(t_start, t_end)-tt)**2)/(200.0**2))
            ws /= ws.sum()
            for t, w in zip(range(t_start, t_end), ws):
                transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
                res1 += w * (
                    pairwise_covariances[t]
                    + np.outer(smoothed_state_means[t],
                               smoothed_state_means[t - 1])
                    - np.outer(transition_offset, smoothed_state_means[t - 1])
                )
                res2 += w * (
                    smoothed_state_covariances[t - 1]
                    + np.outer(smoothed_state_means[t - 1],
                               smoothed_state_means[t - 1])
                )
    #         res[tt-1] = np.linalg.solve(res2.T, res1.T).T
    #         M_inv = np.linalg.solve((res2[0*6:1*6,:] + res2[1*6:2*6,:]).T, res1[2*6:3*6,:].T)
            F_tmp = np.dot(res1, np.linalg.pinv(res2))
            m_inv0 = F_tmp[2*6:3*6,3*6:4*6]
            m_inv1 = F_tmp[2*6:3*6,4*6:5*6]
            m_inv = (m_inv0 + m_inv1) / 2.
            res[tt-1,2*6:3*6,3*6:4*6] = m_inv
            res[tt-1,2*6:3*6,4*6:5*6] = m_inv
        print "time", time.time() - time_start
        return res
    
    kf = KalmanFilter(transition_matrices=F, observation_matrices=H, transition_covariance=Q, observation_covariance=R,
                      initial_state_mean=x0, initial_state_covariance=P0)
    kf = kf.em(Z, n_iter=5, em_vars=['transition_covariance', 'observation_covariance'])
#     kf = kf.em(Z, n_iter=5, em_vars=['transition_matrices'], em_transition_matrix=em_transition_matrix)
    (X_smoothed, P_smoothed) = kf.smooth(Z)
    
    if plotting:
        plt.ion()
        fig = plt.figure()
        for i in range(5):
            plt.subplot(5,2,2*i+1)
            plt.plot(X_smoothed[:,6*i], 'r')
            plt.plot(X_smoothed[:,6*i+1], 'g')
            plt.plot(X_smoothed[:,6*i+2], 'b')
            plt.subplot(5,2,2*i+1+1)
            plt.plot(X_smoothed[:,6*i+3], 'r')
            plt.plot(X_smoothed[:,6*i+4], 'g')
            plt.plot(X_smoothed[:,6*i+5], 'b')
        plt.draw()
        
        fig = plt.figure()
        for i in range(2):
            plt.subplot(2,2,2*i+1)
            plt.plot(Z[:,6*i], 'r')
            plt.plot(Z[:,6*i+1], 'g')
            plt.plot(Z[:,6*i+2], 'b')
            plt.subplot(2,2,2*i+1+1)
            plt.plot(Z[:,6*i+3], 'r')
            plt.plot(Z[:,6*i+4], 'g')
            plt.plot(Z[:,6*i+5], 'b')
        plt.draw()
    
    return X_smoothed

def register_scenes(sim, reg_factory, scene_state):
    sys.stdout.write("registering all scenes... ")
    sys.stdout.flush()
    regs = []
    demos = []
    
#     from rapprentice import plotting_openrave
#     def plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, demo):
#         handles = []
#         handles.append(sim.env.plot3(x_nd, 5, (1,0,0,1)))
#         handles.append(sim.env.plot3(f.transform_points(x_nd), 5, (0,1,0,1)))
#         handles.append(sim.env.plot3(y_md, 5, (0,0,1,1)))
#         handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_nd.min(axis=0) - .1, x_nd.max(axis=0) + .1, xres = .05, yres = .05, zres = .04))
#         aug_traj = demo.aug_traj
#         for lr in 'lr':
#             handles.append(sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 2, (1,0,0)))
#             handles.append(sim.env.drawlinestrip(f.transform_hmats(aug_traj.lr2ee_traj[lr])[:,:3,3], 2, (0,1,0)))
#         print "plot_cb"
#         sim.viewer.Idle()
    
    for action, demo in reg_factory.demos.iteritems():
#         reg = reg_factory.register(demo, scene_state, plotting=5, plot_cb=lambda x_nd, y_md, targ_Nd, corr_nm, wt_n, f: plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, demo))
        reg = reg_factory.register(demo, scene_state)
        regs.append(reg)
        demos.append(demo)
    q_values, regs, demos = zip(*sorted([(reg.f._bending_cost, reg, demo) for (reg, demo) in zip(regs, demos)]))
    print "done"
    
    return regs, demos

def mass_calculate3(lr, robot, aligned_joint_traj):
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    arm = robot.GetManipulator(manip_name)
    
    joint_feedforward = [3.33, 1.16, 0.1, 0.25, 0.133, 0.0727, 0.0727]
    M_joint_inv = np.linalg.inv(np.diag(joint_feedforward))
    M_inv = []

    cur_traj = aligned_joint_traj
    M_inv = []
    Js = []
    for j in range(cur_traj.shape[0]):
        vals = cur_traj[j,:]
        robot.SetDOFValues(vals, arm.GetArmIndices())
        J = np.vstack((arm.CalculateJacobian(), arm.CalculateAngularVelocityJacobian()))
        M_inv.append(J.dot(M_joint_inv).dot(np.transpose(J)))
        Js.append(J)
    M_inv = np.asarray(M_inv)
    Js = np.asarray(Js)
    return M_inv, Js

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

def calculate_masses(lr2arm_traj, robot):
    M_inv = {}
    for lr in 'lr':
        M_inv[lr] = mass_calculate2(lr, robot, lr2arm_traj[lr])
    return M_inv

def calculate_costs(test_aug_traj, robot):
    M_inv = calculate_masses(test_aug_traj.lr2arm_traj, robot)
    
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
        Dts[:,:12,:12] = np.eye(12)
        Dts[:,:6,6:12] = DT
        Dts[:,:6,12:18] = (DT**2)*M_inv[lr][:t_steps] #TODO
        Dts[:,6:12,12:18] = DT*M_inv[lr][:t_steps]
    
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

def mass_calculate(dof_val_mus, dof_val_sigmas, aligned_trajs):
    #Regression code for getting masses
    ipy.embed()
    aligned_trajs = aligned_trajs[:,:,:18]
    n_demos, t_steps, n_dof = aligned_trajs.shape
    
    xdot_trajs = aligned_trajs[:,:,6:12] / DT # TODO have 1/DT somewhere else
    xdotdot_trajs = np.diff(xdot_trajs, axis=1) / DT
    xdotdot_trajs = np.asarray([np.r_[xdotdot_traj, xdotdot_traj[-1][None,:]] for xdotdot_traj in xdotdot_trajs])
    u_trajs = aligned_trajs[:,:,12:18]
    
    lambda_val = 0.1
    alpha = 0.01
    ms = np.zeros((t_steps,6))
    for d in range(6):
        for t in range(t_steps):
            F = []
            Xdot = []
            Xdotdot = []
            w = []
            for i in range(n_demos):
                # time window
                t_start = np.clip(t-100, 0, t_steps)
                t_end = np.clip(t+100+1, 0, t_steps)
                F.append(u_trajs[i,t_start:t_end,d:d+1])
                Xdot.append(xdot_trajs[i,t_start:t_end,d:d+1])
                Xdotdot.append(xdotdot_trajs[i,t_start:t_end,d:d+1])
                w_unnormalized = np.exp(-.5*alpha*((np.arange(t_start, t_end)-t)**2))
                w.append(w_unnormalized/w_unnormalized.sum())
            F = np.concatenate(F)
            F = np.c_[F, np.ones_like(F)]
            Xdot = np.concatenate(Xdot)
            Xdotdot = np.concatenate(Xdotdot)
            w = np.concatenate(w)
            WF = w[:,None] * F
            M = np.linalg.solve(F.T.dot(WF) + lambda_val*np.eye(2), WF.T.dot(Xdotdot))
            ms[t,d] = 1./M[0]
            if d == 0 and t == 300:
                fig = plt.figure()
                plt.scatter(Xdot, F[:,0])
                plt.draw()

    fig = plt.figure()
    for d in range(6):
        ax = fig.add_subplot(6,1,d+1)
        plt.plot(ms[:,d])
        ax.set_yscale('log')
    plt.draw()
    
class ForceAugmentedTrajectory(AugmentedTrajectory):
    def __init__(self, lr2force_traj, lr2x_traj, lr2arm_traj=None, lr2finger_traj=None, lr2ee_traj=None, lr2open_finger_traj=None, lr2close_finger_traj=None):
        super(ForceAugmentedTrajectory, self).__init__(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, 
                                                       lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        for lr2traj in [lr2force_traj, lr2x_traj]:
            if lr2traj is None:
                continue
            for lr in lr2traj.keys():
                assert lr2traj[lr].shape[0] == self.n_steps
        
        self.lr2force_traj = lr2force_traj
        self.lr2x_traj = lr2x_traj
    
    @staticmethod
    def create_from_full_traj(robot, full_traj, lr2force_traj, lr2x_traj, lr2open_finger_traj=None, lr2close_finger_traj=None):
        aug_traj = AugmentedTrajectory.create_from_full_traj(robot, full_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        return ForceAugmentedTrajectory(lr2force_traj, lr2x_traj, lr2arm_traj=aug_traj.lr2arm_traj, lr2finger_traj=aug_traj.lr2finger_traj, lr2ee_traj=aug_traj.lr2ee_traj, lr2open_finger_traj=aug_traj.lr2open_finger_traj, lr2close_finger_traj=aug_traj.lr2close_finger_traj)
    
    def get_resampled_traj(self, timesteps_rs):
        """
        The t step of the resampled trajectory corresponds to the timesteps_rs[t] step of aug_traj. If timesteps_rs[t] is fractional, the appropiate interpolation is used.
        The domain of each element in timesteps_rs is between 0 and aug_traj.n_steps-1 and the length of the resampled trajectory is len(timesteps_rs).
        """
        aug_traj = super(ForceAugmentedTrajectory, self).get_resampled_traj(timesteps_rs)
        lr2force_traj_rs = None if self.lr2force_traj is None else {}
        lr2x_traj_rs = None if self.lr2x_traj is None else {}
        for (lr2traj_rs, self_lr2traj) in [(lr2force_traj_rs, self.lr2force_traj), (lr2x_traj_rs, self.lr2x_traj)]:
            if self_lr2traj is None:
                continue
            for lr in self_lr2traj.keys():
                lr2traj_rs[lr] = math_utils.interp2d(timesteps_rs, np.arange(len(self_lr2traj[lr])), self_lr2traj[lr])
        return ForceAugmentedTrajectory(lr2force_traj_rs, lr2x_traj_rs, lr2arm_traj=aug_traj.lr2arm_traj, lr2finger_traj=aug_traj.lr2finger_traj, lr2ee_traj=aug_traj.lr2ee_traj, 
                                 lr2open_finger_traj=aug_traj.lr2open_finger_traj, lr2close_finger_traj=aug_traj.lr2close_finger_traj)
    
    def get_transformed_traj(self, reg):
        aug_traj = copy.deepcopy(self)
        for lr in self.lr2ee_traj.keys():
            aug_traj.lr2ee_traj[lr] = reg.f.transform_hmats(self.lr2ee_traj[lr])
        for lr in self.lr2force_traj.keys(): # should use the original position for the jacobians
            aug_traj.lr2force_traj[lr] = np.c_[reg.f.transform_vectors(self.lr2ee_traj[lr][:,:3,3], self.lr2force_traj[lr][:,:3]), \
                                               reg.f.transform_vectors(self.lr2ee_traj[lr][:,:3,3], self.lr2force_traj[lr][:,3:])]
        aug_traj.lr2x_traj = {}
        return aug_traj

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
    parser_eval.add_argument("--batch", action="store_true", default=False)

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

    parser_eval.add_argument("--recompute_smoothing", action="store_true", default=False)

    parser_eval.add_argument("--downsample_traj", type=int, default=1, help="downsample demonstration trajectory by this factor")
    parser_eval.add_argument("--max_num_demos", type=int, default=10, help="maximum number of demos to combine")

    args = parser.parse_args()
    return args

def setup_demos(args, robot):
    actions = h5py.File(args.eval.actionfile)
    
    demos = {}
    for action, seg_info in actions.iteritems():
        if 'overhand' in args.eval.actionfile and 'seg00' not in action: continue #TODO
        if 'board3flip0_seg00' == action: continue
        full_cloud = seg_info['cloud_xyz'][()]
        scene_state = SceneState(full_cloud, downsample_size=args.eval.downsample_size)
        lr2force_traj = {}
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
                lr2force_traj[lr] = np.asarray(seg_info["%s_gripper_force"%lr])
        aug_traj = ForceAugmentedTrajectory(lr2force_traj, None, lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        demo = Demonstration(action, scene_state, aug_traj)
        demos[action] = demo

#     flip_wrist_rotations([demo.aug_traj for demo in demos.values()])
#     for action, seg_info in actions.iteritems():
#         aug_traj = demos[action].aug_traj
#         lr2x_traj = {}
#         for lr in 'lr':
#             if "%s_gripper_x"%lr not in seg_info or args.eval.recompute_smoothing:
#                 sys.stdout.write("running Kalman smoothing for demo %s... "%action)
#                 sys.stdout.flush()
#                 lr2M_inv = calculate_masses(aug_traj.lr2arm_traj, robot)
#                 aas = axisAngleFromRotationMatrix(aug_traj.lr2ee_traj[lr][:,:3,:3])
#                 flip_angle_axis_in_place([aas])
#                 Z = np.c_[aug_traj.lr2ee_traj[lr][:,:3,3], 
#                           aas,
#                           aug_traj.lr2force_traj[lr]]
#                 lr2x_traj[lr] = kalman_smoother(Z, lr2M_inv[lr], plotting=False)
#                 del seg_info["%s_gripper_x"%lr]
#                 seg_info["%s_gripper_x"%lr] = lr2x_traj[lr]
#                 print "done"
#             else:
#                 lr2x_traj[lr] = np.asarray(seg_info["%s_gripper_x"%lr])
#         aug_traj.lr2x_traj = lr2x_traj
#     actions.close()
    
#     for demo in demos.values():
#         aug_traj = demo.aug_traj
#         for lr in 'lr':
#             ee_traj = np.empty((len(aug_traj.lr2x_traj[lr]), 4, 4))
#             ee_traj[:] = np.eye(4)
#             ee_traj[:,:3,3] = aug_traj.lr2x_traj[lr][:,:3]
#             ee_traj[:,:3,:3] = rotationMatrixFromAxisAngle(aug_traj.lr2x_traj[lr][:,3:6])
#             force_traj = aug_traj.lr2x_traj[lr][:,3*6:4*6] + aug_traj.lr2x_traj[lr][:,4*6:5*6]
#             aug_traj.lr2ee_traj[lr] = ee_traj
#             aug_traj.lr2force_traj[lr] = force_traj

#             plt.ion()
#             plt.figure()
#             plt.subplot(2,1,1)
#             plt.plot(ee_traj[:,0,3], 'r')
#             plt.plot(ee_traj[:,1,3], 'g')
#             plt.plot(ee_traj[:,2,3], 'b')
#             plt.plot(aug_traj.lr2ee_traj[lr][:,0,3], 'r')
#             plt.plot(aug_traj.lr2ee_traj[lr][:,1,3], 'g')
#             plt.plot(aug_traj.lr2ee_traj[lr][:,2,3], 'b')
#             plt.subplot(2,1,2)
#             plt.plot(force_traj[:,0], 'r')
#             plt.plot(force_traj[:,1], 'g')
#             plt.plot(force_traj[:,2], 'b')
#             plt.plot(aug_traj.lr2force_traj[lr][:,0], 'r')
#             plt.plot(aug_traj.lr2force_traj[lr][:,1], 'g')
#             plt.plot(aug_traj.lr2force_traj[lr][:,2], 'b')
#             plt.draw()
    return demos

def setup_lfd_environment_sim(args):
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
    
    _, seg_info = actions.items()[0]
    init_joint_names = np.asarray(seg_info["joint_states"]["name"])
    init_joint_values = np.asarray(seg_info["joint_states"]["position"][0])
    dof_inds = sim_util.dof_inds_from_name(sim.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    sim.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
    sim_util.reset_arms_to_side(sim)
    actions.close()
    
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
    if args.eval.batch:
        if args.eval.reg_type == 'rpm':
            reg_factory = BatchGpuTpsRpmRegistrationFactory(demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = BatchGpuTpsRpmBijRegistrationFactory(demos, args.eval.actionfile)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(demos)
        elif args.eval.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(demos, n_iter=20, em_iter=1, reg_init=10.0, #TODO
                reg_final=.01, rad_init=.01, rad_final=.001, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                outlierprior=.01, outlierfrac=1e-2)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)

    return reg_factory

def main():
    args = parse_input_args()

    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args)
    demos = setup_demos(args, sim.robot)
    reg_factory = setup_registration(args, demos, sim)
    
#     ipy.embed()
#     for action, demo in demos.iteritems():
#         handles = []
#         handles.append(sim.env.plot3(demo.scene_state.cloud, 2, (0,0,1)))
#         lfd_env.execute_augmented_trajectory(demo.aug_traj, step_viewer=10, interactive=args.interactive)
    
#     import pickle
#     full_cloud = pickle.load(open("data/cloud9.pickle", "rb" ))['cloud']
    # for now, use the cloud of the first demo as the current cloud
    full_cloud = demos.values()[0].scene_state.cloud
    scene_state = demonstration.SceneState(full_cloud, downsample_size=args.eval.downsample_size)

    regs, demos = register_scenes(sim, reg_factory, scene_state)
    
    trajectory_transferer = MultipleDemosPoseTrajectoryTransferer(sim, args.eval.pos_coef, args.eval.rot_coef, args.eval.pos_vel_coef, args.eval.rot_vel_coef, args.eval.force_coef, args.eval.torque_coef, 
                                                                  args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost, 
                                                                  downsample_traj=args.eval.downsample_traj)
    trajectory_transferer.lfd_env = lfd_env
    n_demos = min(args.eval.max_num_demos, len(reg_factory.demos))
    test_aug_traj = trajectory_transferer.transfer(regs[:n_demos], demos[:n_demos], plotting=args.animation, plotting_std_dev=args.std_dev)

    #lr2Cts, lr2cts = calculate_costs(test_aug_traj, sim.robot)
    
    lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)

if __name__ == "__main__":
    main()
