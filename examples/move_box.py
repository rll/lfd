#!/usr/bin/env python

from __future__ import division

import numpy as np
import trajoptpy
from rapprentice import planning, resampling
from core.environment import SimulationEnvironment
from core.simulation_object import XmlSimulationObject, BoxSimulationObject
from core import sim_util

table_height = 0.77
box_half_extents = 0.02
box0_pos = np.r_[.6, -.3, table_height+box_half_extents]
box1_pos = np.r_[.6, 0, table_height+box_half_extents]
sim_objs = []
sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))
sim_objs.append(BoxSimulationObject("box0", box0_pos, [box_half_extents]*3, dynamic=True))
sim_objs.append(BoxSimulationObject("box1", box1_pos, [box_half_extents]*3, dynamic=True))

lfd_env = SimulationEnvironment(sim_objs) # TODO: use only simulation stuff
robot = lfd_env.robot
robot.SetDOFValues([0.25], [robot.GetJoint('torso_lift_joint').GetJointIndex()])
sim_util.reset_arms_to_side(lfd_env)

lfd_env.viewer = trajoptpy.GetViewer(lfd_env.env)
camera_matrix = np.array([[ 0,    1, 0,   0],
                          [-1,    0, 0.5, 0],
                          [ 0.5,  0, 1,   0],
                          [ 2.25, 0, 4.5, 1]])
lfd_env.viewer.SetWindowProp(0,0,1500,1500)
lfd_env.viewer.SetCameraManipulatorMatrix(camera_matrix)

lr = 'r'
manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
ee_link_name = "%s_gripper_tool_frame"%lr
ee_link = robot.GetLink(ee_link_name)
R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
n_steps = 10

def get_move_traj(t_start, t_end, start_fixed):
    hmat_start = np.r_[np.c_[R, t_start], np.c_[0,0,0,1]]
    hmat_end = np.r_[np.c_[R, t_end], np.c_[0,0,0,1]]
    new_hmats = np.asarray(resampling.interp_hmats(np.arange(n_steps), np.r_[0, n_steps-1], [hmat_start, hmat_end]))
    dof_vals = robot.GetManipulator(manip_name).GetArmDOFValues()
    old_traj = np.tile(dof_vals, (n_steps,1))
    
    traj, _, _ = planning.plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, start_fixed=start_fixed, beta_rot=1000.0)
    return traj

move_height = .3
dof_inds = sim_util.dof_inds_from_name(robot, manip_name)

traj = get_move_traj(box0_pos + np.r_[0,0,move_height], box0_pos, False)
lfd_env.execute_trajectory((traj, dof_inds))
lfd_env.close_gripper(lr)

traj = get_move_traj(box0_pos, box0_pos + np.r_[0,0,move_height], True)
lfd_env.execute_trajectory((traj, dof_inds))

traj = get_move_traj(box0_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,move_height], True)
lfd_env.execute_trajectory((traj, dof_inds))

traj = get_move_traj(box1_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,2*box_half_extents+0.005], True)
lfd_env.execute_trajectory((traj, dof_inds))
lfd_env.open_gripper(lr)

lfd_env.viewer.Idle()
