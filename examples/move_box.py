#!/usr/bin/env python

from __future__ import division

import numpy as np
import trajoptpy, openravepy
from rapprentice import planning, resampling
from core.simulation import DynamicSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject
from core import sim_util

table_height = 0.77
box_length = 0.04
box_depth = 0.12
box0_pos = np.r_[.6, -.3, table_height+box_depth/2]
box1_pos = np.r_[.6, 0, table_height+box_depth/2]
sim_objs = []
sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))
sim_objs.append(BoxSimulationObject("box0", box0_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True))
sim_objs.append(BoxSimulationObject("box1", box1_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True))

sim = DynamicSimulationRobotWorld()
sim.add_objects(sim_objs)
sim.robot.SetDOFValues([0.25], [sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
sim_util.reset_arms_to_side(sim)

viewer = trajoptpy.GetViewer(sim.env)
camera_matrix = np.array([[ 0,    1, 0,   0],
                          [-1,    0, 0.5, 0],
                          [ 0.5,  0, 1,   0],
                          [ 2.25, 0, 4.5, 1]])
viewer.SetWindowProp(0,0,1500,1500)
viewer.SetCameraManipulatorMatrix(camera_matrix)

# rotate box0 by 22.5 degrees
bt_box0 = sim.bt_env.GetObjectByName('box0')
T = openravepy.matrixFromAxisAngle(np.array([0,0,np.pi/8]))
T[:3,3] = bt_box0.GetTransform()[:3,3]
bt_box0.SetTransform(T) # SetTransform needs to be used in the Bullet object, not the openrave body
sim.update()

lr = 'r'
manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
ee_link_name = "%s_gripper_tool_frame"%lr
ee_link = sim.robot.GetLink(ee_link_name)
R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
n_steps = 10

def get_move_traj(t_start, t_end, start_fixed):
    hmat_start = np.r_[np.c_[R, t_start], np.c_[0,0,0,1]]
    hmat_end = np.r_[np.c_[R, t_end], np.c_[0,0,0,1]]
    new_hmats = np.asarray(resampling.interp_hmats(np.arange(n_steps), np.r_[0, n_steps-1], [hmat_start, hmat_end]))
    dof_vals = sim.robot.GetManipulator(manip_name).GetArmDOFValues()
    old_traj = np.tile(dof_vals, (n_steps,1))
    
    traj, _, _ = planning.plan_follow_traj(sim.robot, manip_name, ee_link, new_hmats, old_traj, start_fixed=start_fixed, beta_rot=10000.0)
    return traj

move_height = .3
dof_inds = sim_util.dof_inds_from_name(sim.robot, manip_name)

traj = get_move_traj(box0_pos + np.r_[0,0,move_height], box0_pos + np.r_[0,0,box_depth/2-0.02], False)
sim.execute_trajectory((traj, dof_inds))
sim.close_gripper(lr)

traj = get_move_traj(box0_pos + np.r_[0,0,box_depth/2-0.02], box0_pos + np.r_[0,0,move_height], True)
sim.execute_trajectory((traj, dof_inds))

traj = get_move_traj(box0_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,move_height], True)
sim.execute_trajectory((traj, dof_inds))

traj = get_move_traj(box1_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,box_depth+box_depth/2-0.02+0.001], True)
sim.execute_trajectory((traj, dof_inds))
sim.open_gripper(lr)

traj = get_move_traj(box1_pos + np.r_[0,0,box_depth+box_depth/2-0.02+0.002], box1_pos + np.r_[0,0,move_height], True)
sim.execute_trajectory((traj, dof_inds))

sim.settle()
viewer.Idle()
