#!/usr/bin/env python

from __future__ import division

import numpy as np
from core.demonstration import Demonstration, SceneState
from registration.registration import TpsRpmRegistrationFactory
from registration import solver, solver_gpu
from tempfile import mkdtemp
import sys, time
import unittest

class TestRegistration(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        
        def generate_cloud(x_center_pert=0, max_noise=0.02):
            # generates 40 cm by 60 cm cloud with optional pertubation along the x-axis
            grid = np.array(np.meshgrid(np.linspace(-.2,.2,21), np.linspace(-.3,.3,31))).T.reshape((-1,2))
            grid = np.c_[grid, np.zeros(len(grid))]
            cloud = grid + x_center_pert * np.c_[(0.3 - np.abs(grid[:,1]-0))/0.3, np.zeros((len(grid),2))] + (np.random.random((len(grid), 3)) - 0.5) * 2 * max_noise
            return cloud
        
        self.demos = {}
        for x_center_pert in np.arange(-0.1, 0.6, 0.1):
            demo_name = "demo_{}".format(x_center_pert)
            demo_cloud = generate_cloud(x_center_pert=x_center_pert)
            demo_scene_state = SceneState(demo_cloud, downsample_size=0.025)
            demo = Demonstration(demo_name, demo_scene_state, None)
            self.demos[demo_name] = demo
        
        test_cloud = generate_cloud(x_center_pert=0.2)
        self.test_scene_state = SceneState(test_cloud, downsample_size=0.025)
    
    def test_tps_rpm_solvers(self):
        tmp_cachedir = mkdtemp()
        
        reg_factory = TpsRpmRegistrationFactory(self.demos, f_solver_factory=None)
        sys.stdout.write("computing costs: no solver... ")
        sys.stdout.flush()
        start_time = time.time()
        costs = reg_factory.batch_cost(self.test_scene_state)
        print "done in {}s".format(time.time() - start_time)
        
        reg_factory_solver = TpsRpmRegistrationFactory(self.demos, f_solver_factory=solver.TpsSolverFactory(cachedir=tmp_cachedir))
        sys.stdout.write("computing costs: solver... ")
        sys.stdout.flush()
        start_time = time.time()
        costs_solver = reg_factory_solver.batch_cost(self.test_scene_state)
        print "done in {}s".format(time.time() - start_time)
        sys.stdout.write("computing costs: cached solver... ")
        sys.stdout.flush()
        start_time = time.time()
        costs_solver_cached = reg_factory_solver.batch_cost(self.test_scene_state)
        print "done in {}s".format(time.time() - start_time)
        
        reg_factory_gpu = TpsRpmRegistrationFactory(self.demos, f_solver_factory=solver_gpu.TpsGpuSolverFactory(cachedir=tmp_cachedir))
        sys.stdout.write("computing costs: gpu solver... ")
        sys.stdout.flush()
        start_time = time.time()
        costs_gpu = reg_factory_gpu.batch_cost(self.test_scene_state)
        print "done in {}s".format(time.time() - start_time)
        sys.stdout.write("computing costs: cached gpu solver... ")
        sys.stdout.flush()
        start_time = time.time()
        costs_gpu_cached = reg_factory_gpu.batch_cost(self.test_scene_state)
        print "done in {}s".format(time.time() - start_time)
        
        for demo_name in self.demos.keys():
            self.assertTrue(np.allclose(costs[demo_name], costs_solver[demo_name]))
            self.assertTrue(np.allclose(costs[demo_name], costs_solver_cached[demo_name]))
            self.assertTrue(np.allclose(costs[demo_name], costs_gpu[demo_name]))
            self.assertTrue(np.allclose(costs[demo_name], costs_gpu_cached[demo_name]))

if __name__ == '__main__':
    unittest.main()
