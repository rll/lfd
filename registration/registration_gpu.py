from __future__ import division

import numpy as np
from constants import TpsGpuConstant as tpsgc
from registration import Registration, RegistrationFactory
import tps, solver_gpu
import tpsopt
from tpsopt.batchtps import GPUContext, TgtContext

class GpuTpsRpmRegistrationFactory(RegistrationFactory):
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

class GpuTpsRpmBijRegistrationFactory(RegistrationFactory):
    def __init__(self, demos, filename, 
                 n_iter=tpsgc.N_ITER, em_iter=tpsgc.EM_ITER, 
                 reg_init=tpsgc.REG[0], reg_final=tpsgc.REG[1], 
                 rad_init=tpsgc.RAD[0], rad_final=tpsgc.RAD[1], 
                 rot_reg = tpsgc.ROT_REG, 
                 outlierprior = tpsgc.OUTLIER_PRIOR, outlierfrac = tpsgc.OURLIER_FRAC, 
                 prior_fn=None, precompute_fname=None):
        super(GpuTpsRpmBijRegistrationFactory, self).__init__(demos)
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
        self.f_solver_factory = solver_gpu.TpsGpuSolverFactory(tpsgc.MAX_CLD_SIZE, self.n_iter, precompute_fname=precompute_fname)
        self.g_solver_factory = solver_gpu.TpsGpuSolverFactory(tpsgc.MAX_CLD_SIZE, self.n_iter, precompute_fname=precompute_fname)
        
        self.src_ctx = GPUContext(self.regs)
        self.src_ctx.read_h5(filename)
        self.warn_clip_cloud = True
    
    def _clip_cloud(self, cloud):
        if len(cloud) > tpsgc.MAX_CLD_SIZE:
            cloud = cloud[np.random.choice(range(len(cloud)), size=tpsgc.MAX_CLD_SIZE, replace=False)]
        if self.warn_clip_cloud:
            import warnings
            warnings.warn("The cloud has more points than the maximum for GPU and it is being clipped")
            self.warn_clip_cloud = False
        return cloud
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        if self.prior_fn is not None:
            prior_prob_nm = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            prior_prob_nm = None
        x_nd = demo.scene_state.cloud[:,:3]
        y_md = test_scene_state.cloud[:,:3]
        x_nd = self._clip_cloud(x_nd)
        y_md = self._clip_cloud(y_md)
        
        f, g, corr = tps.tps_rpm_bij(x_nd, y_md, 
                                     f_solver_factory=self.f_solver_factory, g_solver_factory=self.g_solver_factory, 
                                     n_iter=self.n_iter, em_iter=self.em_iter, 
                                     reg_init=self.reg_init, reg_final=self.reg_final, 
                                     rad_init=self.rad_init, rad_final=self.rad_final, 
                                     rot_reg=self.rot_reg, 
                                     outlierprior=self.outlierprior, outlierfrac=self.outlierfrac, 
                                     prior_prob_nm=prior_prob_nm)
        
        return Registration(demo, test_scene_state, f, corr, g=g)
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        reg = self.register(demo, test_scene_state, plotting=False, plot_cb=None)
        cost = np.r_[reg.f.getObjective(), reg.g.getObjective()]
        return cost
    
    def batch_cost(self, test_scene_state):
        tgt_ctx = TgtContext(self.src_ctx)
        cloud = test_scene_state.cloud
        cloud = self._clip_cloud(cloud)
        tgt_ctx.set_cld(cloud)
        
        cost_array = tpsopt.batchtps.batch_tps_rpm_bij(self.src_ctx, tgt_ctx, 
                                                       T_init=self.rad_init, T_final=self.rad_final, 
                                                       outlierfrac=self.outlierfrac, outlierprior=self.outlierprior, 
                                                       outliercutoff=tpsgc.OURLIER_CUTOFF, 
                                                       em_iter=self.em_iter, 
                                                       component_cost=True)
        costs = dict(zip(self.src_ctx.seg_names, cost_array))
        return costs
