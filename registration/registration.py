from __future__ import division

import numpy as np
import scipy.spatial.distance as ssd
from constants import TpsConstant as tpsc
import tps

class Registration(object):
    def __init__(self, demo, test_scene_state, f, corr):
        self.demo = demo
        self.test_scene_state = test_scene_state
        self.f = f
        self.corr = corr
    
    def get_objective(self):
        raise NotImplementedError

class TpsRpmRegistration(Registration):
    def __init__(self, demo, test_scene_state, f, corr, rad):
        super(TpsRpmRegistration, self).__init__(demo, test_scene_state, f, corr)
        self.rad = rad
    
    def get_objective(self):
        x_nd = self.demo.scene_state.cloud[:,:3]
        y_md = self.test_scene_state.cloud[:,:3]
        cost = self.get_objective2(x_nd, y_md, self.f, self.corr, self.rad)
        return cost
    
    @staticmethod
    def get_objective2(x_nd, y_md, f, corr_nm, rad):
        """
        Returns the following 5 objectives
        1/n \sum{i=1}^n \sum{j=1}^m corr_nm_ij ||y_md_j - f(x_nd_i)||_2^2
        bend_coef tr(w_ng' K_nn w_ng)
        tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
        rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij log corr_nm_ij
        -rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij
        """
        cost = np.zeros(5)
        dist_nm = ssd.cdist(x_nd, y_md, 'sqeuclidean')
        cost[0] = (corr_nm * dist_nm).sum() / len(x_nd)
        cost[1:3] = f.get_objective()[1:]
        corr_nm = np.reshape(corr_nm, (1,-1))
        nz_corr_nm = corr_nm[corr_nm != 0]
        cost[3] = rad * (nz_corr_nm * np.log(nz_corr_nm)).sum()
        cost[4] = -rad * nz_corr_nm.sum()
        return cost

class TpsRpmBijRegistration(Registration):
    def __init__(self, demo, test_scene_state, f, g, corr, rad):
        super(TpsRpmBijRegistration, self).__init__(demo, test_scene_state, f, corr)
        self.rad = rad
        self.g = g
    
    def get_objective(self):
        raise NotImplementedError


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
    
    Tries to solve the optimization problem
    min_{f,corr_nm} 1/n \sum{i=1}^n \sum{j=1}^m corr_nm_ij ||y_md_j - f(x_nd_i)||_2^2
                    + bend_coef tr(w_ng' K_nn w_ng)
                    + tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
                    + rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij log corr_nm_ij
                    - rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij
    s.t. x_na' w_ng = 0
         1' w_ng = 0
         \sum{i=1}^{n+1} corr_nm_ij = 1
         \sum{j=1}^{m+1} corr_nm_ij = 1
    """
    def __init__(self, demos, 
                 n_iter=tpsc.N_ITER, em_iter=tpsc.EM_ITER, 
                 reg_init=tpsc.REG[0], reg_final=tpsc.REG[1], 
                 rad_init=tpsc.RAD[0], rad_final=tpsc.RAD[1], 
                 rot_reg = tpsc.ROT_REG, 
                 outlierprior = tpsc.OUTLIER_PRIOR, outlierfrac = tpsc.OURLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=None):
        """Inits TpsRpmRegistrationFactory with demonstrations and parameters
        
        Args:
            demos: dict that maps from demonstration name to Demonstration
            n_iter: outer iterations for tps-rpm
            em_iter: inner iterations for tps-rpm
            reg_init/reg_final: regularization on curvature
            rad_init/rad_final: radius (temperature) for correspondence calculation (meters)
            rot_reg: regularization on rotation
            prior_fn: function that takes the demo and test SceneState and returns the prior probability (i.e. NOT cost)
            f_solver_factory: solver factory for forward registration
        
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
        self.f_solver_factory = f_solver_factory
    
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
        
        return TpsRpmRegistration(demo, test_scene_state, f, corr, self.rad_final)
    
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
        cost = reg.f.get_objective()
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
                 f_solver_factory=None, g_solver_factory=None):
        """Inits TpsRpmBijRegistrationFactory with demonstrations and parameters
        
        Args:
            demos: dict that maps from demonstration name to Demonstration
            n_iter: outer iterations for tps-rpm
            em_iter: inner iterations for tps-rpm
            reg_init/reg_final: regularization on curvature
            rad_init/rad_final: radius (temperature) for correspondence calculation (meters)
            rot_reg: regularization on rotation
            prior_fn: function that takes the demo and test SceneState and returns the prior probability (i.e. NOT cost)
            f_solver_factory: solver factory for forward registration
            g_solver_factory: solver factory for backward registration
        
        Note: Pick a T_init that is about 1/10 of the largest square distance of all point pairs
        Tip: Don't cache for the target SolverFactory
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
        self.f_solver_factory = f_solver_factory
        self.g_solver_factory = g_solver_factory
    
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
        
        return TpsRpmBijRegistration(demo, test_scene_state, f, g, corr, self.rad_final)
    
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
        cost = np.r_[reg.f.get_objective(), reg.g.get_objective()]
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
