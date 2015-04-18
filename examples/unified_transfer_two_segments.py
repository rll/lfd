#!/usr/bin/env python

from __future__ import division

import numpy as np

from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.demonstration.demonstration import Demonstration
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import FingerTrajectoryTransferer
from lfd.transfer.registration_transfer_feedback import FeedbackRegistrationAndTrajectoryTransferer
from lfd.transfer.registration_transfer import UnifiedRegistrationAndTrajectoryTransferer
from move_rope import create_augmented_traj, create_rope

def create_initial_rope_demo(env, rope_poss):
    scene_state = env.observe_scene()
    # env.sim.remove_objects([rope_sim_obj])

    pick_pos = rope_poss[0] + .1 * (rope_poss[1] - rope_poss[0])
    drop_pos = rope_poss[3] + .1 * (rope_poss[2] - rope_poss[3]) + np.r_[0, .2, 0]
    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    move_height = .2
    aug_traj = create_augmented_traj(env.sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height)

    demo = Demonstration("rope_demo_1", scene_state, aug_traj)
    return demo

def create_segment_rope_demo(env, rope_poss):
    scene_state = env.observe_scene()
    pick_pos = rope_poss[3] + 0.1 * (rope_poss[2] - rope_poss[3])
    drop_pos = 0.5 * (rope_poss[0] + rope_poss[3])
    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    move_height = 0.2
    aug_traj = create_augmented_traj(env.sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height)
    demo = Demonstration("rope_demo_2", scene_state, aug_traj)
    return demo

def reset_arms(env):
    env.sim.robot.SetDOFValues([0.25], [env.sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
    env.sim.robot.SetDOFValues([1.25], [env.sim.robot.GetJoint('head_tilt_joint').GetJointIndex()]) # move head down so it can see the rope
    sim_util.reset_arms_to_side(env.sim)

def create_two_step_rope_demo(env, rope_poss):
    """Creates two demonstrations of a rope typing procedue"""
    rope_sim_obj = create_rope(rope_poss)
    env.sim.add_objects([rope_sim_obj])
    env.sim.settle()
    # first demo
    demo1 = create_initial_rope_demo(env, rope_poss)
    aug_traj1 = demo1.aug_traj 
    # execute the trajectory
    env.execute_augmented_trajectory(aug_traj1)
    env.sim.settle() 
    reset_arms(env)
    # get the second segment of the demo
    demo2 = create_segment_rope_demo(env, rope_poss) 
    # aug_traj2 = demo2.aug_traj 
    # env.execute_augmented_trajectory(aug_traj2)
    # env.sim.settle() 
    env.sim.remove_objects([rope_sim_obj])
    return (demo1, demo2)
    
def main():
    # define simulation objects
    table_height = 0.77
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))

    # initialize simulation world and environment
    sim = DynamicSimulationRobotWorld()
    sim.add_objects(sim_objs)
    sim.create_viewer()

    sim.robot.SetDOFValues([0.25], [sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
    sim.robot.SetDOFValues([1.25], [sim.robot.GetJoint('head_tilt_joint').GetJointIndex()]) # move head down so it can see the rope
    sim_util.reset_arms_to_side(sim)
    # color_cylinders(cyl_sim_objs)

    env = environment.LfdEnvironment(sim, sim, downsample_size=0.025)

    demo_rope_poss = np.array([[.2, -.2, table_height+0.006],
                               [.8, -.2, table_height+0.006],
                               [.8,  .2, table_height+0.006],
                               [.2,  .2, table_height+0.006]])

    demo1, demo2 = create_two_step_rope_demo(env, demo_rope_poss)

    test_rope_poss = np.array([[.2, -.2, table_height+0.006],
                               [.5, -.4, table_height+0.006],
                               [.8,  .0, table_height+0.006],
                               [.8,  .2, table_height+0.006],
                               [.6,  .0, table_height+0.006],
                               [.4,  .2, table_height+0.006],
                               [.2,  .2, table_height+0.006]])
    test_rope_sim_obj = create_rope(test_rope_poss)
    sim.add_objects([test_rope_sim_obj])
    sim.settle()
    test_scene_state = env.observe_scene()

    reg_factory = TpsRpmRegistrationFactory()
    traj_transferer = FingerTrajectoryTransferer(sim)

    plot_cb = lambda i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad: registration_plot_cb(sim, x_nd, y_md, f)
    # reg_and_traj_transferer = FeedbackRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    test_aug_traj = reg_and_traj_transferer.transfer(demo1, test_scene_state, callback=plot_cb, plotting=True)

    env.execute_augmented_trajectory(test_aug_traj)
    sim.settle()
   
    # transfer second segment
    reset_arms(env)
    test_scene_state = env.observe_scene()

    reg_factory = TpsRpmRegistrationFactory()
    traj_transferer = FingerTrajectoryTransferer(sim)

    plot_cb = lambda i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad: registration_plot_cb(sim, x_nd, y_md, f)
    # reg_and_traj_transferer = FeedbackRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    test_aug_traj = reg_and_traj_transferer.transfer(demo2, test_scene_state, callback=plot_cb, plotting=True)

    env.execute_augmented_trajectory(test_aug_traj)
    

if __name__ == '__main__':
    main()
