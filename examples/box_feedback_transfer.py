#!/usr/bin/env python
from __future__ import division

import numpy as np

from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, BoxRobotSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.demonstration.demonstration import Demonstration, DemonstrationRobot
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import FingerTrajectoryTransferer
# Uncomment the following when using non-test version
from lfd.transfer.feedback_combined import FeedbackRegistrationAndTrajectoryTransferer
# from lfd.transfer.test_feedback_on_f import FeedbackRegistrationAndTrajectoryTransferer
from lfd.transfer.registration_transfer import UnifiedRegistrationAndTrajectoryTransferer
import openravepy
from example_utils import *
from plotting_utils import *

###### Paremeters 
# (TODO): create a file consisting of all parameters setting
INCLUDE_OBSTRUCTION = False
INCLUDE_TIMESTEPS = False

def create_trajectory(env, robot, save_trajectory = False):
    """ Create demonstration trajectory for the robot """
    robot_kinbody = robot.get_bullet_objects()[0].GetKinBody()
    robot_T = robot_kinbody.GetTransform()
    time_steps1 = 25 
    time_steps2 = 10
    total_steps = time_steps1 + time_steps2

    # Let trajectory be a set of poses of the robot (3 degress of freedom)
    # move in quarter circle
    trajectories = np.empty((total_steps, 4, 4))
    d_angle = np.pi / (2 * time_steps1)
    x, y, z, _ = robot_T[:,3]
    r = abs(0 - x)
    target_x = 0
    target_y = y - r
    dx = (target_x - x) / time_steps1
    dy = (target_y - y) / time_steps1
    for i in range(1, time_steps1 + 1):
        angle = d_angle * i
        new_x = x + dx * i
        new_y = y + dy * i
        pose = openravepy.matrixFromAxisAngle([0, 0, angle])
        pose[0, 3] = new_x
        pose[1, 3] = new_y
        pose[2, 3] = z
        trajectories[i-1,:,:] = pose

    # move in straight line
    x, y = target_x, target_y
    target_y = y - 0.15
    target_x = x
    dx = (target_x - x) / time_steps2
    dy = (target_y - y) / time_steps2
    rotation = trajectories[time_steps1 - 1][:3,:3]
    for i in range(1, time_steps2 + 1):
        new_x = x + dx * i
        new_y = y + dy * i
        pose = openravepy.matrixFromPose([1, 0, 0, 0, new_x, new_y, z])
        pose[:3,:3] = rotation 
        trajectories[time_steps1 + i - 1,:,:] = pose
    
    return trajectories

def get_target_pose(env, robot, go_through_hole=False):
    rave_robot = env.sim.env.GetRobots()[0]
    robot_T = rave_robot.GetTransform()
    x, y, z, _ = robot_T[:,3]
    r = abs(0 - x)
    
    target_x = 0
    target_y = y - r
    
    if go_through_hole:
        target_y = y - 0.15 * 2

    target_pose = openravepy.matrixFromAxisAngle([0, 0, np.pi/2])
    target_pose[0, 3] = target_x
    target_pose[1, 3] = target_y
    target_pose[2, 3] = z

    # uncomment if want to view where the target is
    # rave_robot.SetTransform(target_pose)
    # env.sim.viewer.Step()
    # raw_input("! look at target")
    return target_pose
    

def create_demo(env, robot, include_obstruction=False):
    """ Create a demonstration example for the robot """
    robot_kinbody = robot.get_bullet_objects()[0].GetKinBody()

    ### generate trajectory
    trajectory = create_trajectory(env, robot, save_trajectory = True)

    ### Add obstruction pointcloud if include_obstruction is True
    if include_obstruction:
        obstruction_pc = get_all_obstructions_pc(env)
    else:
        obstruction_pc = None

    ### generate sequence of pointcloud from trajectory
    pc_seqs = generate_pc_from_traj(env, robot, robot_kinbody, trajectory, obstruction_pc, plot=False)

    ### generate sequence of rel_pts trajectory
    rel_pts_pc_seq = generate_pc_from_traj(env, robot, robot_kinbody, trajectory, obstruction_pc=None, num_x_points = 2, num_y_points = 2, plot=False)

    demo = DemonstrationRobot("robot_demo_1", pc_seqs, trajectory, rel_pts_pc_seq, obstruction_pc)
    return demo

def color_robot(cyl_sim_objs, color=[1, 0, 0]):
    for sim_obj in cyl_sim_objs:
        # color = np.random.random(3)
        color = np.array(color)
        for bt_obj in sim_obj.get_bullet_objects():
            for link in bt_obj.GetKinBody().GetLinks():
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor(color)
    
def main():
    if INCLUDE_OBSTRUCTION == True and INCLUDE_TIMESTEPS == True:
        raise Exception("Does not support including obstrution pointcloud and timesteps at the same time")

    ##  define simulation objects ##
    sim_objs = []
    table_width = 0.25 #0.85
    table_thickness = 0
    table_x = 0
    table_y = 0
    table_z = 0

    hole_size = 0.03

    obstruction1_length = (table_width - hole_size / 2) / 2
    obstruction1_width = 0.01
    obstruction1_height = 0.03
    obstruction1_x = obstruction1_length + hole_size / 2
    obstruction1_y = 0
    obstruction1_z = 0

    obstruction2_length = obstruction1_length
    obstruction2_width = 0.01
    obstruction2_height = obstruction1_height
    obstruction2_x = -(obstruction2_length + hole_size / 2)
    obstruction2_y = 0
    obstruction2_z = 0

    robot_length = 0.030
    robot_width = 0.008
    robot_height = 0.008
    robot_x = table_x + table_width * 0.50
    robot_y = table_y + table_width * 0.75 + robot_width 
    robot_z = 0

    k = 10

    sim_objs.append(BoxSimulationObject("obstruction1", k*[obstruction1_x, obstruction1_y, obstruction1_z], k*[obstruction1_length, obstruction1_width, obstruction1_height], dynamic=False)) 
    sim_objs.append(BoxSimulationObject("obstruction2", k*[obstruction2_x, obstruction2_y, obstruction2_z], k*[obstruction2_length, obstruction2_width, obstruction2_height], dynamic=False)) 
    robot = BoxRobotSimulationObject("robot", k*[robot_x, robot_y, robot_z], k*[robot_length, robot_width, robot_height], dynamic=False)
    sim_objs.append(robot)

    # initialize simulation world and environment
    sim = DynamicSimulationRobotWorld()
    sim.add_objects(sim_objs)
    sim.create_viewer()
    color_robot([robot], [1, 0, 0])
    env = environment.FeedbackEnvironment(sim, sim)

    sim.viewer.Idle()
    # create demo for the demo robot
    demo = create_demo(env, robot, include_obstruction=INCLUDE_OBSTRUCTION)
    # remove demo robot from the scene
    env.sim.remove_objects([robot])

    # create test robot
    robot_length += 0.070
    robot_width += 0.000
    test_robot = BoxRobotSimulationObject("robot", [robot_x, robot_y, robot_z], [robot_length, robot_width, robot_height], dynamic=False)
    sim.add_objects([test_robot])
    rave_robot = env.sim.env.GetRobots()[0]
    color_robot([test_robot], [0, 0, 1])
    test_scene_state = get_scene_state(env, test_robot, 12, 3, include_obstruction=INCLUDE_OBSTRUCTION)

    target_pose = get_target_pose(env, test_robot, go_through_hole=True)
    rave_robot = env.sim.env.GetRobots()[0]
    rel_pts = get_rel_pts(rave_robot)
    
    reg_factory = TpsRpmRegistrationFactory()

    reg_and_traj_transferer = FeedbackRegistrationAndTrajectoryTransferer(env, reg_factory)
    trajectory = reg_and_traj_transferer.transfer(demo, test_robot, test_scene_state, rel_pts, target_pose=target_pose, timestep_dist = 5, plotting=False, include_timesteps=INCLUDE_TIMESTEPS)
    env.execute_robot_trajectory(rave_robot, trajectory)


if __name__ == '__main__':
    main()
