#!/usr/bin/env python

from __future__ import division

import openravepy

import numpy as np

from lfd.environment.simulation import DynamicSimulation
from lfd.environment.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject, CylinderSimulationObject
from lfd.environment import sim_util


table_height = 0.77
helix_ang0 = 0
helix_ang1 = 4*np.pi
helix_radius = .2
helix_center = np.r_[.6, 0]
helix_height0 = table_height + .15
helix_height1 = table_height + .15 + .3
helix_length = np.linalg.norm(np.r_[(helix_ang1 - helix_ang0) * helix_radius, helix_height1 - helix_height0])
num = np.round(helix_length/.02)
helix_angs = np.linspace(helix_ang0, helix_ang1, num)
helix_heights = np.linspace(helix_height0, helix_height1, num)
init_rope_nodes = np.c_[helix_center + helix_radius * np.c_[np.cos(helix_angs), np.sin(helix_angs)], helix_heights]
rope_params = sim_util.RopeParams()

cyl_radius = 0.025
cyl_height = 0.3
cyl_pos0 = np.r_[.6, helix_radius, table_height + .25]
cyl_pos1 = np.r_[.6, -helix_radius, table_height + .35]

sim_objs = []
sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
sim_objs.append(BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False))
sim_objs.append(RopeSimulationObject("rope", init_rope_nodes, rope_params))
sim_objs.append(CylinderSimulationObject("cyl0", cyl_pos0, cyl_radius, cyl_height, dynamic=True))
sim_objs.append(CylinderSimulationObject("cyl1", cyl_pos1, cyl_radius, cyl_height, dynamic=True))

sim = DynamicSimulation()
sim.add_objects(sim_objs)
sim.create_viewer()

sim.robot.SetDOFValues([0.25], [sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
sim_util.reset_arms_to_side(sim)

# rotate cylinders by 90 deg
for i in range(2):
    bt_cyl = sim.bt_env.GetObjectByName('cyl%d'%i)
    T = openravepy.matrixFromAxisAngle(np.array([np.pi/2,0,0]))
    T[:3,3] = bt_cyl.GetTransform()[:3,3]
    bt_cyl.SetTransform(T) # SetTransform needs to be used in the Bullet object, not the openrave body
sim.update()

sim.settle(max_steps=1000)
sim.viewer.Idle()
