#!/usr/bin/env python

from __future__ import division

import copy
import argparse
from core import sim_util

from core.simulation import DynamicSimulation, ClutterSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject, CoilSimulationObject

from rapprentice import util

from rapprentice import task_execution, rope_initialization
import pdb, time

from mmqe import search

import trajoptpy, openravepy
import os, os.path, numpy as np, h5py
from rapprentice.util import redprint, yellowprint
import atexit
import IPython as ipy
import random
import sys

OBJ_SPACING = 0.1

def rand_pose(x_max, y_max, z_offset=0):
    T = np.eye(4)
    T[0, 3] = x_max * np.random.rand() - x_max / 2
    T[1, 3] = y_max * np.random.rand() - y_max / 2
    T[2, 3] = z_offset

    # rand_d = np.random.rand(3)
    # rand_d = rand_d / np.linalg.norm(rand_d)
    # T[:3, :3] = openravepy.matrixFromAxisAngle(rand_d)[:3, :3]

    return T

def sample_init_state(sim, viewer=None):

    base_x, base_y = sim.container.get_footprint()
    container_pose = sim.container.get_pose()
    z_start = sim.container.get_height() * 2

    objs = [sim.coil] + sim.small_boxes + sim.big_boxes
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

    sim.initialize(init_state, step_viewer=10)
    cld = sim.observe_cloud()
    print 'simulation settled, observation size: {}'.format(cld.shape[0])
    if viewer is not None:
        from rapprentice import plotting_openrave
        handles = []
        handles.append(sim.env.plot3(cld, 3, (0, 1, 0, 1)))
        viewer.Idle()
            
    return init_state, cld

def gen_task_file(N, outf, viewer):

    sim = ClutterSimulationRobotWorld(2, 2)
    sim_util.reset_arms_to_side(sim)

    if viewer:        
        viewer = trajoptpy.GetViewer(sim.env)
        camera_matrix = np.array([[ 0,    1, 0,   0],
                                  [-1,    0, 0.5, 0],
                                  [ 0.5,  0, 1,   0],
                                  [ 2.25, 0, 4.5, 1]])
        viewer.SetWindowProp(0,0,1500,1500)
        viewer.SetCameraManipulatorMatrix(camera_matrix)
    else:
        viewer = None

    if str(N) in outf:
        del outf[N]
    
    for i in range(N):
        g = outf.create_group(str(i))
        state, cloud_xyz = sample_init_state(sim, viewer=viewer)
        state_g = g.create_group('state')
        for n, p in state.iteritems():
            state_g[n] = p
        g['cloud_xyz'] = cloud_xyz

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('outf', type=str)
    parser.add_argument('N', type=int)
    parser.add_argument('--viewer', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    outf = h5py.File(args.outf, 'w')
    gen_task_file(args.N, outf, args.viewer)
    outf.close()
    

        

    
