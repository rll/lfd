#!/usr/bin/env python

from __future__ import division

import numpy as np

from lfd.rapprentice import planning
import lfd.rapprentice.math_utils as mu
from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject, RopeSimulationObject
from lfd.environment import sim_util
from lfd.demonstration.demonstration import AugmentedTrajectory
from lfd.environment import environment

def create_cylinder_grid(cyl_pos0, cyl_pos1, cyl_pos2, cyl_radius, cyl_height):
    sample_grid = np.array(np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))).T.reshape((-1,2))
    cyl_positions = cyl_pos0 + sample_grid[:,0][:,None].dot(cyl_pos1[None,:] - cyl_pos0[None,:]) + sample_grid[:,1][:,None].dot(cyl_pos2[None,:] - cyl_pos0[None,:])
    cyl_sim_objs = []
    for (i,cyl_pos) in enumerate(cyl_positions):
        cyl_sim_objs.append(CylinderSimulationObject("cyl%i"%i, cyl_pos, cyl_radius, cyl_height, dynamic=True))
    return cyl_sim_objs

def color_cylinders(cyl_sim_objs):
    for sim_obj in cyl_sim_objs:
        color = np.random.random(3)
        for bt_obj in sim_obj.get_bullet_objects():
            for link in bt_obj.GetKinBody().GetLinks():
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor(color)

def create_rope(rope_poss, capsule_height=.02):
    rope_pos_dists = np.linalg.norm(np.diff(rope_poss, axis=0), axis=1)
    xp = np.r_[0, np.cumsum(rope_pos_dists/capsule_height)]
    init_rope_nodes = mu.interp2d(np.arange(xp[-1]+1), xp, rope_poss)
    rope_sim_obj = RopeSimulationObject("rope", init_rope_nodes)
    return rope_sim_obj

def create_augmented_traj(robot, pick_pos, drop_pos, pick_R, drop_R, move_height, pos_displacement_per_step=.02):
    pos_traj = np.array([pick_pos + np.r_[0,0,move_height], 
                         pick_pos, 
                         pick_pos + np.r_[0,0,move_height], 
                         drop_pos + np.r_[0,0,move_height], 
                         drop_pos, 
                         drop_pos + np.r_[0,0,move_height]])
    R_traj = np.array([pick_R, pick_R, pick_R, drop_R, drop_R, drop_R])
    ee_traj = np.empty((len(pos_traj), 4, 4))
    ee_traj[:] = np.eye(4)
    ee_traj[:,:3,3] = pos_traj
    ee_traj[:,:3,:3] = R_traj
    open_finger_traj = np.array([False, False, False, False, True, False])
    close_finger_traj = np.array([False, True, False, False, False, False])
    open_finger_value = sim_util.get_binary_gripper_angle(True)
    closed_finger_value = sim_util.get_binary_gripper_angle(False)
    finger_traj = np.array([open_finger_value, open_finger_value, closed_finger_value, closed_finger_value, closed_finger_value, open_finger_value])[:,None]
    
    lr = 'r' # use right arm/gripper
    aug_traj = AugmentedTrajectory(lr2ee_traj={lr: ee_traj}, lr2finger_traj={lr: finger_traj}, lr2open_finger_traj={lr: open_finger_traj}, lr2close_finger_traj={lr: close_finger_traj})
    
    # resample augmented trajectory according to the position displacement
    pos_dists = np.linalg.norm(np.diff(pos_traj, axis=0), axis=1)
    new_times = np.r_[0, np.cumsum(pos_dists/pos_displacement_per_step)]
    timesteps_rs = np.interp(np.arange(new_times[-1]+1), new_times, np.arange(len(new_times)))
    aug_traj_rs = aug_traj.get_resampled_traj(timesteps_rs)
    
    # do motion planning for aug_traj_rs
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    ee_link_name = "%s_gripper_tool_frame"%lr
    ee_link = robot.GetLink(ee_link_name)
    dof_vals = robot.GetManipulator(manip_name).GetArmDOFValues()
    init_traj = np.tile(dof_vals, (aug_traj_rs.n_steps,1))
    arm_traj, _, _ = planning.plan_follow_traj(robot, manip_name, ee_link, aug_traj_rs.lr2ee_traj['r'], init_traj, no_collision_cost_first=True)
    aug_traj_rs.lr2arm_traj[lr] = arm_traj
    
    return aug_traj_rs

def main():
    # define simulation objects
    table_height = 0.77
    cyl_radius = 0.025
    cyl_height = 0.3
    cyl_pos0 = np.r_[.7, -.15, table_height+cyl_height/2]
    cyl_pos1 = np.r_[.7,  .15, table_height+cyl_height/2]
    cyl_pos2 = np.r_[.4, -.15, table_height+cyl_height/2]
    rope_poss = np.array([[.2, -.2, table_height+0.006], 
                          [.8, -.2, table_height+0.006], 
                          [.8,  .2, table_height+0.006], 
                          [.2,  .2, table_height+0.006]])
    
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))
    cyl_sim_objs = create_cylinder_grid(cyl_pos0, cyl_pos1, cyl_pos2, cyl_radius, cyl_height)
    sim_objs.extend(cyl_sim_objs)
    rope_sim_obj = create_rope(rope_poss)
    sim_objs.append(rope_sim_obj)
    
    # initialize simulation world and environment
    sim = DynamicSimulationRobotWorld()
    sim.add_objects(sim_objs)
    sim.create_viewer()
    
    sim.robot.SetDOFValues([0.25], [sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
    sim_util.reset_arms_to_side(sim)
    
    color_cylinders(cyl_sim_objs)
    
    env = environment.LfdEnvironment(sim, sim)
    
    # define augmented trajectory
    pick_pos = rope_poss[0] + .1 * (rope_poss[1] - rope_poss[0])
    drop_pos = rope_poss[3] + .1 * (rope_poss[2] - rope_poss[3]) + np.r_[0, .2, 0]
    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    move_height = .2
    aug_traj = create_augmented_traj(sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height)
    
    env.execute_augmented_trajectory(aug_traj)

if __name__ == '__main__':
    main()
