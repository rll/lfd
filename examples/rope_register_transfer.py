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
from lfd.transfer.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer, DecompRegistrationAndTrajectoryTransferer
from move_rope import create_augmented_traj, create_rope

def create_cylinder(cyl_pos1, cyl_radius, cyl_height):
    sample_grid = np.array(np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))).T.reshape((-1,2))
    cyl_sim_objs = []
    cyl_sim_objs.append(CylinderSimulationObject("obstacle0", cyl_pos1, cyl_radius, cyl_height, dynamic=True))
    return cyl_sim_objs

def color_cylinders(cyl_sim_objs):
    for sim_obj in cyl_sim_objs:
        color = np.random.random(3)
        for bt_obj in sim_obj.get_bullet_objects():
            for link in bt_obj.GetKinBody().GetLinks():
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor(color)


def create_rope_demo(env, rope_poss):
    rope_sim_obj = create_rope(rope_poss)
    env.sim.add_objects([rope_sim_obj])
    env.sim.settle()
    scene_state = env.observe_scene()
    env.sim.remove_objects([rope_sim_obj])

    pick_pos = rope_poss[0] + .1 * (rope_poss[1] - rope_poss[0])
    #drop_pos = pick_pos
    drop_pos = rope_poss[3] + .1 * (rope_poss[2] - rope_poss[3]) + np.r_[0, .2, 0]
    pick_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    drop_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    #pick_R = drop_R
    move_height = .1
    aug_traj = create_augmented_traj(env.sim.robot, pick_pos, drop_pos, pick_R, drop_R, move_height)

    demo = Demonstration("rope_demo", scene_state, aug_traj)
    return demo

def main():
    # define simulation objects
    table_height = 0.77
    cyl_radius = 0.03
    cyl_height = 0.2
    cyl_pos = np.r_[.25, 0, table_height+cyl_height/2]
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

    demo_rope_poss = np.array([[.2, -.2, table_height+0.006],
                               [.8, -.2, table_height+0.006],
                               [.8,  .2, table_height+0.006],
                               [.2,  .2, table_height+0.006]])
    demo = create_rope_demo(env, demo_rope_poss)

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
    reg_and_traj_transferer = DecompRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    test_aug_traj = reg_and_traj_transferer.transfer(demo, test_scene_state, callback=plot_cb, plotting=True)

    env.execute_augmented_trajectory(test_aug_traj)

if __name__ == '__main__':
    main()
