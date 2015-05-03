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
from lfd.transfer.registration_transfer_feedback import FeedbackRegistrationAndTrajectoryTransferer
from lfd.transfer.registration_transfer import UnifiedRegistrationAndTrajectoryTransferer
from move_rope import create_augmented_traj, create_rope
from lfd.transfer.test_trajopt import plan_trajopt
import openravepy

def color_robot(cyl_sim_objs, color=[1, 0, 0]):
    for sim_obj in cyl_sim_objs:
        # color = np.random.random(3)
        color = np.array(color)
        for bt_obj in sim_obj.get_bullet_objects():
            for link in bt_obj.GetKinBody().GetLinks():
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor(color)
    
def main():
    # define simulation objects
    sim_objs = []
    table_width = 0.25 #0.85
    table_thickness = 0.05
    table_x = 0
    table_y = 0
    table_z = 0

    hole_size = 0.03

    obstruction1_length = (table_width - hole_size / 2) / 2
    obstruction1_width = 0.01
    obstruction1_height = 0.03
    obstruction1_x = obstruction1_length + hole_size / 2
    obstruction1_y = 0
    obstruction1_z = table_thickness + obstruction1_height

    obstruction2_length = obstruction1_length
    obstruction2_width = 0.01
    obstruction2_height = obstruction1_height
    obstruction2_x = -(obstruction2_length + hole_size / 2)
    obstruction2_y = 0
    obstruction2_z = table_thickness + obstruction2_height

    # on the bottom right corner. a 2D robot with three dimensions of freedom
    robot_length = 0.030
    robot_width = 0.008
    robot_height = 0.008
    robot_x = table_x + table_width * 0.50
    robot_y = table_y + table_width * 0.75 + robot_width 
    robot_z = table_z + table_thickness + robot_height

    k = 10

    sim_objs.append(BoxSimulationObject("table", k*[table_x, table_y, table_z], k*[table_width, table_width, table_thickness], dynamic=False))
    sim_objs.append(BoxSimulationObject("obstruction1", k*[obstruction1_x, obstruction1_y, obstruction1_z], k*[obstruction1_length, obstruction1_width, obstruction1_height], dynamic=False)) 
    sim_objs.append(BoxSimulationObject("obstruction2", k*[obstruction2_x, obstruction2_y, obstruction2_z], k*[obstruction2_length, obstruction2_width, obstruction2_height], dynamic=False)) 
    # import pdb; pdb.set_trace()
    robot = BoxRobotSimulationObject("robot", k*[robot_x, robot_y, robot_z], k*[robot_length, robot_width, robot_height], dynamic=False)
    # robot = BoxSimulationObject("robot", k*[robot_x, robot_y, robot_z], k*[robot_length, robot_width, robot_height], dynamic=True)
    sim_objs.append(robot)

    # initialize simulation world and environment
    sim = DynamicSimulationRobotWorld()
    sim.add_objects(sim_objs)
    sim.create_viewer()
    color_robot([robot], [1, 0, 0])
    env = environment.FeedbackEnvironment(sim, sim)
    plan_trajopt(env, robot, plotting=True)
    

    sim.viewer.Idle()


    # env.execute_augmented_trajectory(test_aug_traj)

if __name__ == '__main__':
    main()
