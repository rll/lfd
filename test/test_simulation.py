#!/usr/bin/env python

from __future__ import division

import numpy as np
import openravepy, trajoptpy
from core.simulation import DynamicSimulation
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
        
        self.sim = DynamicSimulation()
        self.sim.add_objects(sim_objs)
        self.sim.robot.SetDOFValues([0.25], [self.sim.robot.GetJoint('torso_lift_joint').GetJointIndex()])
        sim_util.reset_arms_to_side(self.sim)
        
        # rotate cylinders by 90 deg
        for i in range(2):
            bt_cyl = self.sim.bt_env.GetObjectByName('cyl%d'%i)
            T = openravepy.matrixFromAxisAngle(np.array([np.pi/2,0,0]))
            T[:3,3] = bt_cyl.GetTransform()[:3,3]
            bt_cyl.SetTransform(T) # SetTransform needs to be used in the Bullet object, not the openrave body
        self.sim.update()
        viewer = trajoptpy.GetViewer(self.sim.env)
        camera_matrix = np.array([[ 0,    1, 0,   0],
                                  [-1,    0, 0.5, 0],
                                  [ 0.5,  0, 1,   0],
                                  [ 2.25, 0, 4.5, 1]])
        viewer.SetWindowProp(0,0,1500,1500)
        viewer.SetCameraManipulatorMatrix(camera_matrix)
        viewer.Step()
    
    def test_reproducibility(self):
        sim_state0 = self.sim.get_state()
        
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state1 = self.sim.get_state()
        
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state2 = self.sim.get_state()
        
        self.assertArrayDictEqual(sim_state1[1], sim_state2[1])
    
    def test_viewer_side_effects(self):
        """
        Check if stepping the viewer has side effects in the simulation
        """
        sim_state0 = self.sim.get_state()
        
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000, step_viewer=0)
        sim_state1 = self.sim.get_state()
        
        # create viewer
        viewer = trajoptpy.GetViewer(self.sim.env)
        
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000, step_viewer=10)
        sim_state2 = self.sim.get_state()
        
        self.assertArrayDictEqual(sim_state1[1], sim_state2[1])
    
    def test_remove_sim_obj(self):
        sim_state0 = self.sim.get_state()
        
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state1 = self.sim.get_state()
        
        rope = [sim_obj for sim_obj in self.sim.sim_objs if isinstance(sim_obj, RopeSimulationObject)][0]
        box = BoxSimulationObject("box", [0]*3, [.1]*3, dynamic=False)

        self.sim.remove_objects([rope])
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state2 = self.sim.get_state() # this adds another rope that has the same properties as rope
        
        self.sim.add_objects([box])
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state3 = self.sim.get_state() # this removes the recently added box
        
        rope = [sim_obj for sim_obj in self.sim.sim_objs if isinstance(sim_obj, RopeSimulationObject)][0]
        
        self.sim.remove_objects([rope])
        self.sim.add_objects([box])
        self.sim.set_state(sim_state0)
        self.sim.settle(max_steps=1000)
        sim_state4 = self.sim.get_state()
        
        self.assertArrayDictEqual(sim_state1[1], sim_state2[1])
        self.assertArrayDictEqual(sim_state1[1], sim_state3[1])
        self.assertArrayDictEqual(sim_state1[1], sim_state4[1])
    
    def assertArrayDictEqual(self, d0, d1):
        self.assertSetEqual(set(d0.keys()), set(d1.keys()))
        for (k, v0) in d0.iteritems():
            v1 = d1[k]
            self.assertTrue(np.all(v0 == v1))

if __name__ == '__main__':
    unittest.main()
