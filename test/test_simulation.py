#!/usr/bin/env python

from __future__ import division

import numpy as np
import openravepy
from core.environment import SimulationEnvironment
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject, RopeSimulationObject
from core import sim_util
import unittest

class TestSimulation(unittest.TestCase):
    def setUp(self):
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
        
        self.lfd_env = SimulationEnvironment(sim_objs) # TODO: use only simulation stuff
        robot = self.lfd_env.robot
        robot.SetDOFValues([0.25], [robot.GetJoint('torso_lift_joint').GetJointIndex()])
        sim_util.reset_arms_to_side(self.lfd_env)
        
        # rotate cylinders by 90 deg
        for i in range(2):
            bt_cyl = self.lfd_env.bt_env.GetObjectByName('cyl%d'%i)
            T = openravepy.matrixFromAxisAngle(np.array([np.pi/2,0,0]))
            T[:3,3] = bt_cyl.GetTransform()[:3,3]
            bt_cyl.SetTransform(T) # SetTransform needs to be used in the Bullet object, not the openrave body
        self.lfd_env._update_rave()

    def test_reproducibility(self):
        lfd_state0 = self.lfd_env.get_state()
        
        self.lfd_env.set_state(lfd_state0)
        self.lfd_env.settle(max_steps=1000)
        lfd_state1 = self.lfd_env.get_state()
        
        self.lfd_env.set_state(lfd_state0)
        self.lfd_env.settle(max_steps=1000)
        lfd_state2 = self.lfd_env.get_state()
        
        for obj_state1, obj_state2 in zip(lfd_state1[0], lfd_state2[0]):
            self.assertTrue(np.all(obj_state1 == obj_state2))
        for dof_limit_state1, dof_limit_state2 in zip(lfd_state1[1], lfd_state2[1]):
            self.assertTrue(np.all(dof_limit_state1 == dof_limit_state2))
        self.assertTrue(np.all(lfd_state1[2] == lfd_state2[2]))

if __name__ == '__main__':
    unittest.main()
