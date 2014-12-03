#!/usr/bin/env python

from __future__ import division

import numpy as np

from lfd.rapprentice import resampling
from lfd.transfer import planning
from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.demonstration.demonstration import Demonstration, SceneState
from lfd.demonstration.demonstration import AugmentedTrajectory
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import FingerTrajectoryTransferer
from lfd.transfer.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer, DecompRegistrationAndTrajectoryTransferer
from move_rope import create_augmented_traj, create_rope
# from move_box import get_move_traj

table_height = 0.77

R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

def generate_cloud(x_center_pert=0, max_noise=0.02):
    # generates 40 cm by 60 cm cloud with optional pertubation along the x-axis
    # grid = np.array(np.meshgrid(np.linspace(-.2,.2,21), np.linspace(-.3,.3,31))).T.reshape((-1,2))
    grid = np.array(np.meshgrid(np.linspace(-.2,.2,5), np.linspace(-.3,.3,8))).T.reshape((-1,2))
    grid = np.c_[grid, np.zeros(len(grid))] + np.array([.6, 0, table_height+max_noise])
    cloud = grid + x_center_pert * np.c_[(0.3 - np.abs(grid[:,1]-0))/0.3, np.zeros((len(grid),2))] + (np.random.random((len(grid), 3)) - 0.5) * 2 * max_noise
    return cloud

def get_move_traj(env, robot, t_start, t_end, start_fixed):
    lr = 'r'
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    ee_link_name = "%s_gripper_tool_frame"%lr
    ee_link = robot.GetLink(ee_link_name)
    n_steps = 10

    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    hmat_start = np.r_[np.c_[R, t_start], np.c_[0,0,0,1]]
    hmat_end = np.r_[np.c_[R, t_end], np.c_[0,0,0,1]]
    new_hmats = np.asarray(resampling.interp_hmats(np.arange(n_steps), np.r_[0, n_steps-1], [hmat_start, hmat_end]))
    dof_vals = robot.GetManipulator(manip_name).GetArmDOFValues()
    old_traj = np.tile(dof_vals, (n_steps,1))

    traj, _, _ = planning.plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, start_fixed=start_fixed, beta_rot=10000.0)
    aug_traj = AugmentedTrajectory(lr2arm_traj={lr: traj}) 
    # import ipdb; ipdb.set_trace()
    return aug_traj



def create_cylinder(cyl_pos1, cyl_radius, cyl_height):
    sample_grid = np.array(np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))).T.reshape((-1,2))
    cyl_sim_objs = []
    cyl_sim_objs.append(CylinderSimulationObject("cyl0", cyl_pos1, cyl_radius, cyl_height, dynamic=True))
    return cyl_sim_objs

def color_cylinders(cyl_sim_objs):
    for sim_obj in cyl_sim_objs:
        color = np.random.random(3)
        for bt_obj in sim_obj.get_bullet_objects():
            for link in bt_obj.GetKinBody().GetLinks():
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor(color)

def main():
    # define simulation objects
    # import ipdb; ipdb.set_trace()
    cyl_radius = 0.1
    cyl_height = 0.2
    cyl_pos = np.r_[.7,0, table_height+cyl_height/2]
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))
    cyl_sim_objs = create_cylinder(cyl_pos, cyl_radius, cyl_height)
    sim_objs.extend(cyl_sim_objs)

    # initialize simulation world and environment
    sim = DynamicSimulationRobotWorld()
    sim.add_objects(sim_objs)
    sim.create_viewer()

    sim.robot.SetDOFValues([0.25], [sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
    sim.robot.SetDOFValues([1.25], [sim.robot.GetJoint('head_tilt_joint').GetJointIndex()]) # move head down so it can see the rope
    sim_util.reset_arms_to_side(sim)
    color_cylinders(cyl_sim_objs)

    env = environment.LfdEnvironment(sim, sim, downsample_size=0.025)


    demos = {}
    x_center_pert = 0.0
    demo_name = "demo_{}".format(x_center_pert)
    demo_cloud = generate_cloud(x_center_pert=x_center_pert)
    demo_scene_state = SceneState(demo_cloud, downsample_size=0.025)


    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

    # table_height = 0.77
    box_length = 0.04
    box_depth = 0.12
    box0_pos = np.r_[.6, -.3, table_height+box_depth/2]
    box1_pos = np.r_[.6, .3, table_height+box_depth/2]
    move_height = .3
    # aug_traj = get_move_traj(env, sim.robot, box0_pos + np.r_[0,0,box_depth/2-0.02], box1_pos + np.r_[0,0,box_depth+box_depth/2-0.02+0.001], False)
    # aug_traj = get_move_traj(env,sim.robot, pick_R, drop_R, False)

    # aug_traj = create_augmented_traj(sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height=0)
    # pick_pos = rope_poss[0] + .1 * (rope_poss[1] - rope_poss[0])
    # drop_pos = rope_poss[3] + .1 * (rope_poss[2] - rope_poss[3]) + np.r_[0, .2, 0]
    pick_pos = box0_pos
    drop_pos = box1_pos
    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    aug_traj = create_augmented_traj(sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height=0.01)
    
    demo = Demonstration(demo_name, demo_scene_state, aug_traj)
    demos[demo_name] = demo

    test_scene_state = SceneState(demo_cloud, downsample_size=0.025)

    plot_cb = lambda i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad: registration_plot_cb(sim, x_nd, y_md, f)

    reg_factory = TpsRpmRegistrationFactory(demos, f_solver_factory=None)
    regs = reg_factory.batch_register(test_scene_state, callback=plot_cb)
    # regs = reg_factory.register(demo, test_scene_state, callback=plot_cb)

    # test_scene_state = env.observe_scene()

    reg_factory = TpsRpmRegistrationFactory()
    traj_transferer = FingerTrajectoryTransferer(sim)

    plot_cb = lambda i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad: registration_plot_cb(sim, x_nd, y_md, f)
    reg_and_traj_transferer = DecompRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    test_aug_traj = reg_and_traj_transferer.transfer(demo, test_scene_state, callback=plot_cb, plotting=True)

    env.execute_augmented_trajectory(test_aug_traj)

if __name__ == '__main__':
    main()
