#!/usr/bin/env python

from __future__ import division

import numpy as np
from lfd.demonstration.demonstration import Demonstration, SceneState
from lfd.registration.registration import TpsRpmRegistration, TpsRpmRegistrationFactory
from lfd.registration import tps, solver
from lfd.registration import _has_cuda
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
        for x_center_pert in np.arange(0.1, 0.4, 0.1):
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
        
        reg_factory_solver = TpsRpmRegistrationFactory(self.demos, f_solver_factory=solver.CpuTpsSolverFactory(cachedir=tmp_cachedir))
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
        
        if _has_cuda:
            reg_factory_gpu = TpsRpmRegistrationFactory(self.demos, f_solver_factory=solver.GpuTpsSolverFactory(cachedir=tmp_cachedir))
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
        else:
            print "couldn't run GPU tests since the GPU is not configured properly"
        
        for demo_name in self.demos.keys():
            self.assertTrue(np.allclose(costs[demo_name], costs_solver[demo_name]))
            self.assertTrue(np.allclose(costs[demo_name], costs_solver_cached[demo_name]))
            if _has_cuda:
                self.assertTrue(np.allclose(costs[demo_name], costs_gpu[demo_name]))
                self.assertTrue(np.allclose(costs[demo_name], costs_gpu_cached[demo_name]))
    
    def test_tps_objective(self):
        reg_factory = TpsRpmRegistrationFactory({}, f_solver_factory=solver.CpuTpsSolverFactory(use_cache=False))
        reg = reg_factory.register(self.demos.values()[0], self.test_scene_state)
        
        x_na = reg.f.x_na
        y_ng = reg.f.y_ng
        wt_n = reg.f.wt_n
        rot_coef = reg.f.rot_coef
        bend_coef = reg.f.bend_coef
        
        # code from tps_fit3
        n,d = x_na.shape
        
        K_nn = tps.tps_kernel_matrix(x_na)
        Q = np.c_[np.ones((n,1)), x_na, K_nn]
        rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else np.asarray(rot_coef)
        A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
        
        WQ = wt_n[:,None] * Q
        QWQ = Q.T.dot(WQ)
        H = QWQ
        H[d+1:,d+1:] += bend_coef * K_nn
        H[1:d+1, 1:d+1] += np.diag(rot_coefs)
        
        f = -WQ.T.dot(y_ng)
        f[1:d+1,0:d] -= np.diag(rot_coefs)
        
        # optimum point
        theta = np.r_[reg.f.trans_g[None,:], reg.f.lin_ag, reg.f.w_ng]
        
        # equality constraint
        self.assertTrue(np.allclose(A.dot(theta), np.zeros((4,3))))
        # objective
        obj = np.trace(theta.T.dot(H.dot(theta))) + 2*np.trace(f.T.dot(theta)) \
        + np.trace(y_ng.T.dot(wt_n[:,None]*y_ng)) + rot_coefs.sum() # constant
        self.assertTrue(np.allclose(obj, reg.f.get_objective().sum()))

    def test_tpsrpm_objective_monotonicity(self):
        n_iter = 10
        em_iter = 10
        reg_factory = TpsRpmRegistrationFactory(n_iter=n_iter, em_iter=em_iter, f_solver_factory=solver.AutoTpsSolverFactory(use_cache=False))
        
        objs = np.zeros((n_iter, em_iter))
        def callback(i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad):
            objs[i, i_em] = TpsRpmRegistration.get_objective2(x_nd, y_md, f, corr_nm, rad).sum()
        
        reg = reg_factory.register(self.demos.values()[0], self.test_scene_state, callback=callback)
        print np.diff(objs, axis=1) <= 0 # TODO assert when monotonicity is more robust

if __name__ == '__main__':
    unittest.main()
