from __future__ import division

import numpy as np
from constants import TpsConstant as tpsc
import tps, solver

class Registration(object):
    def __init__(self, demo, test_scene_state, f, corr, g=None):
        self.demo = demo
        self.test_scene_state = test_scene_state
        self.f = f
        self.corr = corr
        self.g = g

class RegistrationFactory(object):
    def __init__(self, demos):
        """Inits RegistrationFactory with demonstrations
        
        Args:
            demos: dict that maps from demonstration name to Demonstration
        """
        self.demos = demos
        
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        """Registers demonstration scene onto the test scene
        
        Args:
            demo: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
            plotting: 0 means don't plot. integer n means plot every n iterations
            plot_cb: plotting callback with arguments: x_nd, y_md, xtarg_nd, corr_nm, wt_n, f
        
        Returns:
            A Registration
        """
        raise NotImplementedError

    def batch_register(self, test_scene_state):
        registrations = {}
        for name, demo in self.demos.iteritems():
            registrations[name] = self.register(demo, test_scene_state)
        return registrations
    
    def cost(self, demo, test_scene_state):
        """Gets costs of registering the demonstration scene onto the 
        test scene
        
        Args:
            demo: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
        
        Returns:
            A 1-dimensional numpy.array containing the partial costs used for 
            the registration; the sum of these is the objective used for the 
            registration. The exact definition of these partial costs is given 
            by the derived classes.
        """
        raise NotImplementedError

    def batch_cost(self, test_scene_state):
        costs = {}
        for name, demo in self.demos.iteritems():
            costs[name] = self.cost(demo, test_scene_state)
        return costs

class TpsRpmRegistrationFactory(RegistrationFactory):
    """
    As in:
        H. Chui and A. Rangarajan, "A new point matching algorithm for non-rigid registration," Computer Vision and Image Understanding, vol. 89, no. 2, pp. 114-141, 2003.
    """
    def __init__(self, demos, 
                 n_iter=tpsc.N_ITER, em_iter=tpsc.EM_ITER, 
                 reg_init=tpsc.REG[0], reg_final=tpsc.REG[1], 
                 rad_init=tpsc.RAD[0], rad_final=tpsc.RAD[1], 
                 rot_reg = tpsc.ROT_REG, 
                 outlierprior = tpsc.OUTLIER_PRIOR, outlierfrac = tpsc.OURLIER_FRAC, 
                 prior_fn=None, 
                 use_solver=False):
        """Inits TpsRpmRegistrationFactory with demonstrations and parameters
        
        Args:
            demos: dict that maps from demonstration name to Demonstration
            n_iter: outer iterations for tps-rpm
            em_iter: inner iterations for tps-rpm
            reg_init/reg_final: regularization on curvature
            rad_init/rad_final: radius (temperature) for correspondence calculation (meters)
            rot_reg: regularization on rotation
            prior_fn: function that takes the demo and test SceneState and returns the prior probability (i.e. NOT cost)
            use_solver: whether to use SolverFactory
        
        Note: Pick a T_init that is about 1/10 of the largest square distance of all point pairs
        """
        super(TpsRpmRegistrationFactory, self).__init__(demos)
        self.n_iter = n_iter
        self.em_iter = em_iter
        self.reg_init = reg_init
        self.reg_final = reg_final
        self.rad_init = rad_init
        self.rad_final = rad_final
        self.rot_reg = rot_reg
        self.outlierprior = outlierprior
        self.outlierfrac = outlierfrac
        self.prior_fn = prior_fn
        
        if use_solver:
            self.f_solver_factory = solver.TpsSolverFactory()
        else:
            self.f_solver_factory = None
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        if self.prior_fn is not None:
            prior_prob_nm = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            prior_prob_nm = None
        x_nd = demo.scene_state.cloud[:,:3]
        y_md = test_scene_state.cloud[:,:3]
        
        f, corr = tps.tps_rpm(x_nd, y_md, 
                              f_solver_factory=self.f_solver_factory, 
                              n_iter=self.n_iter, em_iter=self.em_iter, 
                              reg_init=self.reg_init, reg_final=self.reg_final, 
                              rad_init=self.rad_init, rad_final=self.rad_final, 
                              rot_reg=self.rot_reg, 
                              outlierprior=self.outlierprior, outlierfrac=self.outlierfrac, 
                              prior_prob_nm=prior_prob_nm, plotting=plotting, plot_cb=plot_cb)
        
        return Registration(demo, test_scene_state, f, corr)
    
    def cost(self, demo, test_scene_state):
        """Gets the costs of the thin plate spline objective of the 
        resulting registration
        
        Args:
            demo: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
        
        Returns:
            A 1-dimensional numpy.array containing the residual, bending and 
            rotation cost, each already premultiplied by the respective 
            coefficients.
        """
        reg = self.register(demo, test_scene_state, plotting=False, plot_cb=None)
        cost = reg.f.getObjective()
        return cost

class TpsRpmBijRegistrationFactory(RegistrationFactory):
    """
    As in:
        J. Schulman, J. Ho, C. Lee, and P. Abbeel, "Learning from Demonstrations through the Use of Non-
        Rigid Registration," in Proceedings of the 16th International Symposium on Robotics Research 
        (ISRR), 2013.
    """
    def __init__(self, demos, 
                 n_iter=tpsc.N_ITER, em_iter=tpsc.EM_ITER, 
                 reg_init=tpsc.REG[0], reg_final=tpsc.REG[1], 
                 rad_init=tpsc.RAD[0], rad_final=tpsc.RAD[1], 
                 rot_reg = tpsc.ROT_REG, 
                 outlierprior = tpsc.OUTLIER_PRIOR, outlierfrac = tpsc.OURLIER_FRAC, 
                 prior_fn=None, 
                 use_solver=False):
        """Inits TpsRpmBijRegistrationFactory with demonstrations and parameters
        
        Args:
            demos: dict that maps from demonstration name to Demonstration
            n_iter: outer iterations for tps-rpm
            em_iter: inner iterations for tps-rpm
            reg_init/reg_final: regularization on curvature
            rad_init/rad_final: radius (temperature) for correspondence calculation (meters)
            rot_reg: regularization on rotation
            prior_fn: function that takes the demo and test SceneState and returns the prior probability (i.e. NOT cost)
            use_solver: whether to use SolverFactory
        
        Note: Pick a T_init that is about 1/10 of the largest square distance of all point pairs
        """
        super(TpsRpmBijRegistrationFactory, self).__init__(demos)
        self.n_iter = n_iter
        self.em_iter = em_iter
        self.reg_init = reg_init
        self.reg_final = reg_final
        self.rad_init = rad_init
        self.rad_final = rad_final
        self.rot_reg = rot_reg
        self.outlierprior = outlierprior
        self.outlierfrac = outlierfrac
        self.prior_fn = prior_fn
        
        if use_solver:
            self.f_solver_factory = solver.TpsSolverFactory()
            self.g_solver_factory = solver.TpsSolverFactory()
        else:
            self.f_solver_factory = None
            self.g_solver_factory = None
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        if self.prior_fn is not None:
            prior_prob_nm = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            prior_prob_nm = None
        x_nd = demo.scene_state.cloud[:,:3]
        y_md = test_scene_state.cloud[:,:3]
        
        f, g, corr = tps.tps_rpm_bij(x_nd, y_md, 
                                     f_solver_factory=self.f_solver_factory, g_solver_factory=self.g_solver_factory, 
                                     n_iter=self.n_iter, em_iter=self.em_iter, 
                                     reg_init=self.reg_init, reg_final=self.reg_final, 
                                     rad_init=self.rad_init, rad_final=self.rad_final, 
                                     rot_reg=self.rot_reg, 
                                     outlierprior=self.outlierprior, outlierfrac=self.outlierfrac, 
                                     prior_prob_nm=prior_prob_nm, plotting=plotting, plot_cb=plot_cb)
        
        return Registration(demo, test_scene_state, f, corr, g=g)
    
    def cost(self, demo, test_scene_state):
        """Gets the costs of the forward and backward thin plate spline 
        objective of the resulting registration
        
        Args:
            demo: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
        
        Returns:
            A 1-dimensional numpy.array containing the residual, bending and 
            rotation cost of the forward and backward spline, each already 
            premultiplied by the respective coefficients.
        """
        reg = self.register(demo, test_scene_state, plotting=False, plot_cb=None)
        cost = np.r_[reg.f.getObjective(), reg.g.getObjective()]
        return cost

class TpsSegmentRegistrationFactory(RegistrationFactory):
    def __init__(self, demos):
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        raise NotImplementedError
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        raise NotImplementedError

class TpsnRpmRegistrationFactory(RegistrationFactory):
    """
    TPS-RPM using normals information
    """
    def __init__(self, demos):
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        raise NotImplementedError
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        raise NotImplementedError
