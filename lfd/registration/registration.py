from __future__ import division

import numpy as np
import scipy.spatial.distance as ssd
import settings
import tps
import solver
import lfd.registration
if lfd.registration._has_cuda:
    from lfd.tpsopt.batchtps import batch_tps_rpm_bij, GPUContext, TgtContext

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
        r"""Returns the following 5 objectives:
        
            - :math:`\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} ||y_j - f(x_i)||_2^2`
            - :math:`\lambda Tr(A^\top K A)`
            - :math:`Tr((B - I) R (B - I))`
            - :math:`\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \log m_{ij}`
            - :math:`-\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij}`
        """
        cost = np.zeros(5)
        xwarped_nd = f.transform_points(x_nd)
        dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
        n = len(x_nd)
        cost[0] = (corr_nm * dist_nm).sum() / n
        cost[1:3] = f.get_objective()[1:]
        corr_nm = np.reshape(corr_nm, (1,-1))
        nz_corr_nm = corr_nm[corr_nm != 0]
        cost[3] = (2*rad / n) * (nz_corr_nm * np.log(nz_corr_nm)).sum()
        cost[4] = -(2*rad / n) * nz_corr_nm.sum()
        return cost


class TpsRpmBijRegistration(Registration):
    def __init__(self, demo, test_scene_state, f, g, corr, rad):
        super(TpsRpmBijRegistration, self).__init__(demo, test_scene_state, f, corr)
        self.rad = rad
        self.g = g
    
    def get_objective(self):
        x_nd = self.demo.scene_state.cloud[:,:3]
        y_md = self.test_scene_state.cloud[:,:3]
        cost = self.get_objective2(x_nd, y_md, self.f, self.g, self.corr, self.rad)
        return cost
    
    @staticmethod
    def get_objective2(x_nd, y_md, f, g, corr_nm, rad):
        r"""Returns the following 10 objectives:
        
            - :math:`\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} ||y_j - f(x_i)||_2^2`
            - :math:`\lambda Tr(A_f^\top K A_f)`
            - :math:`Tr((B_f - I) R (B_f - I))`
            - :math:`\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \log m_{ij}`
            - :math:`-\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij}`
            - :math:`\frac{1}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij} ||x_i - g(y_j)||_2^2`
            - :math:`\lambda Tr(A_g^\top K A_g)`
            - :math:`Tr((B_g - I) R (B_g - I))`
            - :math:`\frac{2T}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij} \log m_{ij}`
            - :math:`-\frac{2T}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij}`
        """
        cost = np.r_[TpsRpmRegistration.get_objective2(x_nd, y_md, f, corr_nm, rad), 
                     TpsRpmRegistration.get_objective2(y_md, x_nd, g, corr_nm.T, rad)]
        return cost


class TpsnRpmRegistration(Registration):
    def __init__(self, demo, test_scene_state, f, corr, x_ld, u_rd, z_rd, y_md, v_sd, z_sd, rad, radn, bend_coef, rot_coef):
        super(TpsRpmRegistration, self).__init__(demo, test_scene_state, f, corr)
        self.x_ld = x_ld
        self.u_rd = u_rd
        self.z_rd = z_rd
        self.y_md = y_md
        self.v_sd = v_sd
        self.z_sd = z_sd
        self.rad = rad
        self.radn = radn
        self.bend_coef = bend_coef
        self.rot_coef = rot_coef 
    
    def get_objective(self):
        x_nd = self.demo.scene_state.cloud[:,:3]
        y_md = self.test_scene_state.cloud[:,:3]
        # TODO: fill x_ld, u_rd, z_rd, y_md, v_sd, z_sd
        cost = self.get_objective2(x_ld, u_rd, z_rd, y_md, v_sd, z_sd, self.f, self.corr_lm, self.corr_rs, self.rad, self.radn, self.bend_coef, self.rot_coef)
        return cost

    @staticmethod
    def get_objective2(x_ld, u_rd, z_rd, y_md, v_sd, z_sd, f, corr_lm, corr_rs, rad, radn, bend_coef, rot_coef):
        r"""Returns the following 5 objectives:
        
            - :math:`\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} ||y_j - f(x_i)||_2^2`
            - :math:`\lambda Tr(A^\top K A)`
            - :math:`Tr((B - I) R (B - I))`
            - :math:`\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \log m_{ij}`
            - :math:`-\frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij}`
        """
        cost = np.zeros(8)
        xwarped_ld = f.transform_points()
        uwarped_rd = f.transform_vectors()
        zwarped_rd = f.transform_points(z_rd)

        beta_r = np.linalg.norm(uwarped_rd, axis=1)

        dist_lm = ssd.cdist(xwarped_ld, y_md, 'sqeuclidean')
        dist_rs = ssd.cdist(uwarped_rd / beta_r[:,None], v_sd, 'sqeuclidean')
        site_dist_rs = ssd.cdist(zwarped_rd, z_sd, 'sqeuclidean')
        prior_prob_rs = np.exp( -site_dist_rs / (2*rad) )

        l = len(x_ld)
        r = len(u_rd)
        # point matching cost
        cost[0] = (corr_lm * dist_lm).sum() / l
        # normal matching cost
        cost[1] = (corr_rs * dist_rs).sum() / r

        # bending cost
        cost[2] = f.compute_bending_energy(bend_coef=bend_coef)

        # rotation cost
        cost[3] = f.compute_rotation_reg(rot_coef=rot_coef)

        # point entropy
        corr_lm = np.reshape(corr_lm, (1,-1))
        nz_corr_lm = corr_lm[corr_lm != 0]
        cost[4] = (2*rad / l) * (nz_corr_lm * np.log(nz_corr_lm)).sum()
        cost[5] = -(2*rad / l) * nz_corr_lm.sum()

        # normal entropy
        corr_rs = np.reshape(corr_rs, (1,-1))
        nz_corr_rs = corr_rs[corr_rs != 0]
        nz_site_dist_rs = site_dist_rs[corr_rs != 0]
        cost[6] = (2*radn / r) * (nz_corr_rs * np.log(nz_corr_rs / nz_site_dist_rs)).sum()
        cost[7] = -(2*radn / r) * nz_corr_rs.sum()
        return cost


class RegistrationFactory(object):
    def __init__(self, demos=None):
        """Inits RegistrationFactory with demonstrations
        
        Args:
            demos: dict that maps from demonstration name to Demonstration. 
                This is used by batch_registration and batch_cost.
        """
        if demos is None:
            self.demos = {}
        else:
            self.demos = demos
        
    def register(self, demo, test_scene_state, callback=None):
        """Registers demonstration scene onto the test scene
        
        Args:
            demo: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
            callback: callback function; the derived classes define the 
                arguments of the functoin
        
        Returns:
            A Registration
        """
        raise NotImplementedError

    def batch_register(self, test_scene_state, callback=None):
        """Registers every demonstration scene in demos onto the test scene
        
        Returns:
            A dict that maps from the demonstration names that are in demos 
            to the Registration
        
        Note:
            Derived classes might ignore the argument callback
        """
        registrations = {}
        for name, demo in self.demos.iteritems():
            registrations[name] = self.register(demo, test_scene_state, callback=callback)
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
        """Gets costs of every demonstration scene in demos registered onto 
        the test scene
        
        Returns:
            A dict that maps from the demonstration names that are in demos 
            to the numpy.array of partial cost
        """
        costs = {}
        for name, demo in self.demos.iteritems():
            costs[name] = self.cost(demo, test_scene_state)
        return costs


class TpsRpmRegistrationFactory(RegistrationFactory):
    r"""As in:
    
    H. Chui and A. Rangarajan, "A new point matching algorithm for non-rigid registration," Computer Vision and Image Understanding, vol. 89, no. 2, pp. 114-141, 2003.
    
    Tries to solve the optimization problem
    
    .. math::
        :nowrap:

        \begin{align*}
            & \min_{f, M} 
                & \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} ||y_j - f(x_i)||_2^2
                + \lambda Tr(A^\top K A)
                + Tr((B - I) R (B - I)) \\
                && + \frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \log m_{ij}
                - \frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \\
            & \text{subject to} 
                & X^\top A = 0, 1^\top A = 0 \\
                && \sum_{i=1}^{n+1} m_{ij} = 1, \sum_{j=1}^{m+1} m_{ij} = 1, m_{ij} \geq 0 \\
        \end{align*}
    """
    def __init__(self, demos=None, 
                 n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
                 reg_init=settings.REG[0], reg_final=settings.REG[1], 
                 rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
                 rot_reg=settings.ROT_REG, 
                 outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OUTLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=solver.AutoTpsSolverFactory()):
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
        
        Note:
            Pick a T_init that is about 1/10 of the largest square distance of all point pairs.
        """
        super(TpsRpmRegistrationFactory, self).__init__(demos=demos)
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
    
    def register(self, demo, test_scene_state, callback=None):
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
                              prior_prob_nm=prior_prob_nm, callback=callback)
        
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
        reg = self.register(demo, test_scene_state, callback=None)
        cost = reg.f.get_objective()
        return cost


class TpsRpmBijRegistrationFactory(RegistrationFactory):
    r"""As in:
    
    J. Schulman, J. Ho, C. Lee, and P. Abbeel, "Learning from Demonstrations through the Use of Non-Rigid Registration," in Proceedings of the 16th International Symposium on Robotics Research (ISRR), 2013.
    
    Tries to solve the optimization problem
    
    .. math::
        :nowrap:

        \begin{align*}
            & \min_{f, M} 
                & \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} ||y_j - f(x_i)||_2^2
                + \lambda Tr(A_f^\top K A_f)
                + Tr((B_f - I) R (B_f - I)) \\
                && + \frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \log m_{ij}
                - \frac{2T}{n} \sum_{i=1}^n \sum_{j=1}^m m_{ij} \\
                && + \frac{1}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij} ||x_i - g(y_j)||_2^2
                + \lambda Tr(A_g^\top K A_g)
                + Tr((B_g - I) R (B_g - I)) \\
                && + \frac{2T}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij} \log m_{ij}
                - \frac{2T}{m} \sum_{j=1}^m \sum_{i=1}^n m_{ij} \\
            & \text{subject to} 
                & X^\top A_f = 0, 1^\top A_f = 0 \\
                && Y^\top A_g = 0, 1^\top A_g = 0 \\
                && \sum_{i=1}^{n+1} m_{ij} = 1, \sum_{j=1}^{m+1} m_{ij} = 1, m_{ij} \geq 0 \\
        \end{align*}
    """
    def __init__(self, demos=None, 
                 n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
                 reg_init=settings.REG[0], reg_final=settings.REG[1], 
                 rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
                 rot_reg=settings.ROT_REG, 
                 outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OUTLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=solver.AutoTpsSolverFactory(), 
                 g_solver_factory=solver.AutoTpsSolverFactory(use_cache=False)):
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
        
        Note:
            Pick a T_init that is about 1/10 of the largest square distance of all point pairs.
            You might not want to cache for the target SolverFactory.
        """
        super(TpsRpmBijRegistrationFactory, self).__init__(demos=demos)
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
    
    def register(self, demo, test_scene_state, callback=None):
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
                                     prior_prob_nm=prior_prob_nm, callback=callback)
        
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
        reg = self.register(demo, test_scene_state, callback=None)
        cost = np.r_[reg.f.get_objective(), reg.g.get_objective()]
        return cost


class BatchGpuTpsRpmRegistrationFactory(TpsRpmRegistrationFactory):
    """
    Similar to TpsRpmRegistrationFactory but batch_register and batch_cost are computed in batch using the GPU
    """
    def __init__(self, demos):
        if not lfd.registration._has_cuda:
            raise NotImplementedError("CUDA not installed")
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, callback=None):
        raise NotImplementedError
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        raise NotImplementedError


class BatchGpuTpsRpmBijRegistrationFactory(TpsRpmBijRegistrationFactory):
    """
    Similar to TpsRpmBijRegistrationFactory but batch_register and batch_cost are computed in batch using the GPU
    """
    def __init__(self, demos, actionfile=None, 
                 n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
                 reg_init=settings.REG[0], reg_final=settings.REG[1], 
                 rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
                 rot_reg=settings.ROT_REG, 
                 outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OUTLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=solver.AutoTpsSolverFactory(), 
                 g_solver_factory=solver.AutoTpsSolverFactory(use_cache=False)):
        if not lfd.registration._has_cuda:
            raise NotImplementedError("CUDA not installed")
        super(BatchGpuTpsRpmBijRegistrationFactory, self).__init__(demos=demos, 
                                                              n_iter=n_iter, em_iter=em_iter, 
                                                              reg_init=reg_init, reg_final=reg_final, 
                                                              rad_init=rad_init, rad_final=rad_final, 
                                                              rot_reg=rot_reg, 
                                                              outlierprior=outlierprior, outlierfrac=outlierfrac, 
                                                              prior_fn=prior_fn, 
                                                              f_solver_factory=f_solver_factory, g_solver_factory=g_solver_factory)

        self.actionfile = actionfile
        if self.actionfile:
            self.bend_coefs = tps.loglinspace(self.reg_init, self.reg_final, self.n_iter)
            self.src_ctx = GPUContext(self.bend_coefs)
            self.src_ctx.read_h5(actionfile)
        self.warn_clip_cloud = True
    
    def _clip_cloud(self, cloud):
        if len(cloud) > settings.MAX_CLD_SIZE:
            cloud = cloud[np.random.choice(range(len(cloud)), size=settings.MAX_CLD_SIZE, replace=False)]
        if self.warn_clip_cloud:
            import warnings
            warnings.warn("The cloud has more points than the maximum for GPU and it is being clipped")
            self.warn_clip_cloud = False
        return cloud
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        if not(self.actionfile):
            raise ValueError('No actionfile provided for gpu context')
        tgt_ctx = TgtContext(self.src_ctx)
        cloud = test_scene_state.cloud
        cloud = self._clip_cloud(cloud)
        tgt_ctx.set_cld(cloud)
        
        cost_array = batch_tps_rpm_bij(self.src_ctx, tgt_ctx,
                                       T_init=self.rad_init, T_final=self.rad_final, 
                                       outlierfrac=self.outlierfrac, outlierprior=self.outlierprior, 
                                       outliercutoff=settings.OUTLIER_CUTOFF, 
                                       em_iter=self.em_iter, 
                                       component_cost=True)
        costs = dict(zip(self.src_ctx.seg_names, cost_array))
        return costs


class TpsSegmentRegistrationFactory(RegistrationFactory):
    def __init__(self, demos):
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, callback=None):
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
    def __init__(self, demos=None, 
                 n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
                 reg_init=settings.REG[0], reg_final=settings.REG[1], 
                 rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
                 rot_reg=settings.ROT_REG, 
                 outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OUTLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=solver.AutoTpsSolverFactory()):
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, callback=None):
        if self.prior_fn is not None:
            prior_prob_nm = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            prior_prob_nm = None
        x_nd = demo.scene_state.cloud[:,:3]
        y_md = test_scene_state.cloud[:,:3]
        
        f, corr_lm, corr_rs = tps_experimental.tpsn_rpm(x_ld, u_rd, z_rd, y_md, v_sd, z_sd, 
                                                        n_iter=self.n_iter, em_iter=self.em_iter, 
                                                        reg_init=self.reg_init, reg_final=self.reg_final, 
                                                        rad_init=self.rad_init, rad_final=self.rad_final, 
                                                        radn_init=self.radn_init, radn_final=self.radn_final, 
                                                        nu_init=nu_init, nu_final=nu_final, 
                                                        rot_reg=self.rot_reg, 
                                                        outlierprior=self.outlierprior, outlierfrac=self.outlierfrac, 
                                                        callback=callback)

        return TpsnRpmRegistration(demo, test_scene_state, f, corr, self.rad_final)
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError
