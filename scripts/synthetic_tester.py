#!/usr/bin/env python

from __future__ import division

from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.environment.robot_world import RealRobotWorld
from lfd.demonstration.demonstration import Demonstration, SceneState, AugmentedTrajectory
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import PoseTrajectoryTransferer
import scipy.spatial.distance as ssd
import sys
import colorsys
import pickle
import datetime
from lfd.lfmd.analyze_data import dof_val_cost, align_trajs
from lfd.registration import tps_experimental, hc
from lfd.rapprentice import plotting_openrave, clouds
import IPython as ipy
import matplotlib.pyplot as plt

import argparse

from lfd.rapprentice import berkeley_pr2
try:
    from lfd.rapprentice import pr2_trajectories, PR2
    import rospy
except:
    print "Couldn't import ros stuff"

# import cloudprocpy, trajoptpy, openravepy

import os, numpy as np, h5py, time
from numpy import asarray
import importlib




def generate_2d_cloud(x_length, y_length, step):
    grid = np.array(np.meshgrid(np.arange(0, x_length, step), np.arange(0, y_length, step))).T.reshape((-1,2))
    return grid

def generate_box_cloud(x_length, y_length, z_length, step=0.03, max_noise=0.005):
    bottom_cloud = generate_2d_cloud(x_length, y_length, step)
    sizex_cloud = generate_2d_cloud(x_length, z_length, step)
    sizey_cloud = generate_2d_cloud(y_length, z_length, step)
    box = np.r_[np.c_[bottom_cloud, np.zeros(len(bottom_cloud))],
                np.c_[sizex_cloud[:,0], np.zeros(len(sizex_cloud)), sizex_cloud[:,1]],
                np.c_[sizex_cloud[:,0], y_length * np.ones(len(sizex_cloud)), sizex_cloud[:,1]],
                np.c_[np.zeros(len(sizey_cloud)), sizey_cloud],
                np.c_[x_length * np.ones(len(sizey_cloud)), sizey_cloud]]
    box[:,0] -= x_length/2
    box[:,1] -= y_length/2
    if max_noise != 0:
        box += (np.random.random((len(box), 3)) - 0.5) * 2 * max_noise
    return box

def generate_obj_cloud(x_length, y_length, z_length, step=0.03, max_noise=0.005):
    obj = generate_box_cloud(x_length, y_length, z_length, step=step, max_noise=max_noise)
    obj[:,2] = z_length - obj[:,2]
    return obj

def generate_scene_cloud_color(obj_pos=None, obj_pos1 = None):
    box_cloud = generate_box_cloud(0.30, 0.50, 0.15)
    box_color = np.ones(box_cloud.shape) * np.array([.5,.1,.1])
    obj_cloud = generate_obj_cloud(0.05, 0.05, 0.30)
    # 
    if obj_pos is not None:
        obj_cloud += obj_pos

    obj_color = np.ones(obj_cloud.shape) * np.array([0,0,1])
    # 
    cloud = np.r_[box_cloud, obj_cloud]
    color = np.r_[box_color, obj_color]
    if obj_pos1 is not None:
        obj_cloud1 = generate_obj_cloud(0.05, 0.05, 0.20)
        obj_cloud1 += obj_pos1
        obj_color1 = np.ones(obj_cloud1.shape) * np.array([0,0,0])
        cloud = np.r_[cloud, obj_cloud1]
        color = np.r_[color, obj_color1]
    return cloud, color

def l2_plotter(x, colors, sim):
    handles = []
    # # x_ld = f.x_la
    # xwarped_nd = f.transform_points()
    # handles.append(sim.env.plot3(xwarped_nd, 5, (0,0,1)))
    # # handles.append(sim.env.plot3(xwarped_nd, 5, (0,1,0)))
    # handles.append(sim.env.plot3(y_md, 5, (1,0,0)))
    # # handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_ld.min(axis=0) - .1, x_ld.max(axis=0) + .1, xres=.1, yres=.1, zres=.1))
    # sim.viewer.Step()
    if colors == None:
        colors = (0,0,1)
    handles.append(sim.env.plot3(x, 5, colors))
    sim.viewer.Step()
    return handles

def multi_l2_callback(f_k, y_md, p_ktd, sim, colors):
    handles = []
    if len(colors) > len(f_k):
        colors = colors[0:len(f_k)]
    for f, p_td, color in zip(f_k, p_ktd, colors):
        x_ld = f.x_la
        xwarped_nd = f.transform_points(x_ld)
        # handles.append(sim.env.plot3(x_ld, 3, color))
        handles.append(sim.env.plot3(xwarped_nd, 4, color))
        pwarped_td = f.transform_points(p_td)
        handles.append(sim.env.drawlinestrip(pwarped_td, 2, color))
        sim.viewer.Step()
        raw_input('disping')
#         handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_ld.min(axis=0) - .1, x_ld.max(axis=0) + .1, xres=.05, yres=.05, zres=.04, color=color))
    if y_md is not None:
        handles.append(sim.env.plot3(y_md, 6, (0,1,0)))
    sim.viewer.Idle()
    return handles

def show_image(rgb, name):
    import cv2
    cv2.imshow(name, rgb)
    cv2.waitKey()


def main():
   
    demos = [None]*5
    i = 0
    handles = []
    offset = np.array([0.5, 0.0, 0.75])
    bottle_vals = [np.array([0.05, 0.15, 0]), np.array([0.05, -0.15, 0]), np.array([-0.05, 0.15, 0]), np.array([-0.05, -0.15, 0]), np.array([0.0, 0.0, 0])]
    hammer_vals = [np.array([0.05, -0.15, 0]), None, np.array([0.05, -0.15, 0]), np.array([0.05, -0.15, 0]), np.array([0.05, -0.15, 0])]

    x_kld = []
    allcolors = []
    for bottle_val, hammer_val in zip(bottle_vals, hammer_vals):
        cloud, color = generate_scene_cloud_color(bottle_val, hammer_val)
        cloud += offset
        x_kld.append(cloud)
        allcolors.append(color)
        i+=1
   
    
    y_md = x_kld[1]
    targetcolors = allcolors[1]

    handles = []
    
    demo_colors = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, max(len(demos), 4), endpoint=False)]

    targetcolors = None
    prior_prob_lms = [None]*len(x_kld)
    
    p_ktd = []   
    for x_ld, i, bottle_val, color in zip(x_kld, range(5), bottle_vals, allcolors):
        p_ktd.append(np.c_[np.r_[[(bottle_val + offset)[:2]]*500], np.arange(1.05, 1.45, 0.4/500)])
        print("CREATING")
    p_ktd = np.array(p_ktd)
    
    traj_rad = 0.2
    w_kt = []
    for x_ld, p_td in zip(x_kld, p_ktd):
        dist_tl = ssd.cdist(p_td, x_ld, 'sqeuclidean')
        dist_t = np.min(dist_tl, axis=1)
        w_kt.append(np.exp(-dist_t / (traj_rad**2)))
    w_kt = np.asarray(w_kt)
    w_t = w_kt.mean(axis=0)
    f_k = []
    for x_ld in x_kld:
        from lfd.rapprentice import clouds
        f = tps_experimental.ThinPlateSpline(x_ld, clouds.downsample(x_ld, .05))
        f_k.append(f)
    
    cov_coef = 0.6
    handles = []
    f_k = tps_experimental.pairwise_tps_rpm_cov_nn(x_kld, y_md, p_ktd, f_init_k=f_k, n_iter=2, allcolors = allcolors, targetcolors = targetcolors, prior_prob_lms=prior_prob_lms, cov_coef=cov_coef, w_t=w_t, w_pc=None, reg_init=10, reg_final=1, rad_init=1, rad_final=.1, rot_reg=np.r_[100, 100, 100], 
                                               callback=l2_plotter, args=(), multi_callback=multi_l2_callback, multi_args=())
        # f_k = tps_experimental.pairwise_tps_l2_cov(x_kld, y_md, p_ktd, f_init_k=f_k, n_iter=2, cov_coef=cov_coef, w_t=w_t, reg_init=100, reg_final=10, rad_init=1, rad_final=.1, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                                   # callback=l2_callback, args=(sim,), multi_callback=multi_l2_callback, multi_args=(sim, demo_colors))
    return f_k


if __name__ == '__main__':
    main()