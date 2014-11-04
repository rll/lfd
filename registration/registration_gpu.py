from __future__ import division

import numpy as np
from constants import TpsGpuConstant as tpsgc
from registration import Registration, TpsRpmRegistrationFactory, TpsRpmBijRegistrationFactory
import tps, tpsopt
from tpsopt.batchtps import GPUContext, TgtContext

class BatchGpuTpsRpmRegistrationFactory(TpsRpmRegistrationFactory):
    """
    Similar to TpsRpmRegistrationFactory but batch_register and batch_cost are computed in batch using the GPU
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

class BatchGpuTpsRpmBijRegistrationFactory(TpsRpmBijRegistrationFactory):
    """
    Similar to TpsRpmBijRegistrationFactory but batch_register and batch_cost are computed in batch using the GPU
    TODO: this class is broken and needs to be implemented
    """
    def __init__(self, demos, 
                 n_iter=tpsgc.N_ITER, em_iter=tpsgc.EM_ITER, 
                 reg_init=tpsgc.REG[0], reg_final=tpsgc.REG[1], 
                 rad_init=tpsgc.RAD[0], rad_final=tpsgc.RAD[1], 
                 rot_reg = tpsgc.ROT_REG, 
                 outlierprior = tpsgc.OUTLIER_PRIOR, outlierfrac = tpsgc.OURLIER_FRAC, 
                 prior_fn=None, 
                 f_solver_factory=None, g_solver_factory=None):
        super(BatchGpuTpsRpmBijRegistrationFactory, self).__init__(demos=demos, 
                                                              n_iter=n_iter, em_iter=em_iter, 
                                                              reg_init=reg_init, reg_final=reg_final, 
                                                              rad_init=rad_init, rad_final=rad_final, 
                                                              rot_reg=rot_reg, 
                                                              outlierprior=outlierprior, outlierfrac=outlierfrac, 
                                                              prior_fn=prior_fn, 
                                                              f_solver_factory=f_solver_factory, g_solver_factory=g_solver_factory)
        raise NotImplementedError
        regs = tps.loglinspace(reg_init, reg_final, n_iter)
        self.src_ctx = GPUContext(regs)
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
    
    def batch_register(self, test_scene_state):
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        tgt_ctx = TgtContext(self.src_ctx)
        cloud = test_scene_state.cloud
        cloud = self._clip_cloud(cloud)
        tgt_ctx.set_cld(cloud)
        
        cost_array = tpsopt.batchtps.batch_tps_rpm_bij(self.src_ctx, tgt_ctx, 
                                                       T_init=self.rad_init, T_final=self.rad_final, 
                                                       outlierfrac=self.outlierfrac, outlierprior=self.outlierprior, 
                                                       outliercutoff=tpsgc.OUTLIER_CUTOFF, 
                                                       em_iter=self.em_iter, 
                                                       component_cost=True)
        costs = dict(zip(self.src_ctx.seg_names, cost_array))
        return costs
