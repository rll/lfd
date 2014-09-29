from __future__ import division

from constants import EXACT_LAMBDA, DEFAULT_LAMBDA, N_ITER_EXACT, N_ITER_CHEAP
import numpy as np
from rapprentice import registration

import tpsopt
from constants import BEND_COEF_DIGITS, MAX_CLD_SIZE
from tpsopt.tps import tps_kernel_matrix
from tpsopt.registration import loglinspace

try:
    from tpsopt.batchtps import GPUContext, TgtContext, SrcContext, batch_tps_rpm_bij
    from tpsopt.transformations import EmptySolver, TPSSolver
except:
    print 'Skipping GPU Import - Failure'


import IPython as ipy

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
            demo_scene_state: Demonstration which has the demonstration scene
            test_scene_state: SceneState of the test scene
        
        Returns:
            A correspondence matrix and a Transformation
        """
        raise NotImplementedError

    def batch_register(self, test_scene_state):
        registrations = {}
        for name, demo in self.demos.iteritems():
            registrations[name] = self.register(demo, test_scene_state)
        return registrations
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError

    def batch_cost(self, test_scene_state):
        costs = {}
        for name, demo in self.demos.iteritems():
            costs[name] = self.cost(demo, test_scene_state)
        return costs


class TpsRpmBijRegistrationFactory(RegistrationFactory):
    """
    As in:
        J. Schulman, J. Ho, C. Lee, and P. Abbeel, "Learning from Demonstrations through the Use of Non-
        Rigid Registration," in Proceedings of the 16th International Symposium on Robotics Research 
        (ISRR), 2013.
    """
    def __init__(self, demos, actionfile=None, n_iter=N_ITER_EXACT, em_iter=1, reg_init=EXACT_LAMBDA[0], 
        reg_final=EXACT_LAMBDA[1], rad_init=.1, rad_final=.005, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
        outlierprior=.1, outlierfrac=1e-2, cost_type='bending', prior_fn=None):
        """
        TODO: do something better for default parameters and write comment
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
        self.cost_type = cost_type
        self.prior_fn = prior_fn

        self.actionfile = actionfile
        if self.actionfile:
            self.bend_coefs = np.around(loglinspace(DEFAULT_LAMBDA[0], DEFAULT_LAMBDA[1], N_ITER_CHEAP), BEND_COEF_DIGITS)
            self.exact_bend_coefs = np.around(loglinspace(EXACT_LAMBDA[0], EXACT_LAMBDA[1], N_ITER_EXACT), BEND_COEF_DIGITS)
            self.f_empty_solver = EmptySolver(MAX_CLD_SIZE, self.exact_bend_coefs)
            self.g_empty_solver = EmptySolver(MAX_CLD_SIZE, self.exact_bend_coefs)
            self.src_ctx = GPUContext(self.bend_coefs)
            self.src_ctx.read_h5(actionfile)

       
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        """
        TODO: use em_iter
        """
        if self.prior_fn is not None:
            vis_cost_xy = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            vis_cost_xy = None
        old_cloud = demo.scene_state.cloud
        new_cloud = test_scene_state.cloud
        x_nd = old_cloud[:,:3]
        y_md = new_cloud[:,:3]
        scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        scaled_y_md, targ_params = registration.unit_boxify(y_md)
        x_weights = np.ones(len(old_cloud)) * 1.0/len(old_cloud)
        (f,g), corr = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md,
                                    x_weights = x_weights,
                                    n_iter = self.n_iter,
                                    reg_init = self.reg_init,
                                    reg_final = self.reg_final,
                                    rad_init = self.rad_init,
                                    rad_final = self.rad_final,
                                    rot_reg = self.rot_reg,
                                    outlierprior = self.outlierprior,
                                    outlierfrac = self.outlierfrac,
                                    vis_cost_xy = vis_cost_xy,
                                    return_corr = True,
                                    plotting = plotting,
                                    plot_cb = plot_cb)
        bending_cost = registration.tps_reg_cost(f)
        f = registration.unscale_tps(f, src_params, targ_params)
        f._bending_cost = bending_cost # TODO: do this properly
        return Registration(demo, test_scene_state, f, corr, g=g)

    # def batch_register(self, test_scene_state):
    #     # TODO Dylan
    #     raise NotImplementedError

    def cost(self, demo, test_scene_state):
        # TODO Dylan
        if self.cost_type == 'bending':
            reg = self.register(demo, test_scene_state, plotting=False, plot_cb=None)
            return reg.f._bending_cost
        else:
            raise NotImplementedError

    def batch_cost(self, test_scene_state):
        if not(self.actionfile):
            raise ValueError('No actionfile provided for gpu context')
        tgt_ctx = TgtContext(self.src_ctx)
        cloud = test_scene_state.cloud
        if len(cloud) > MAX_CLD_SIZE:
            cloud = cloud[np.random.choice(range(len(cloud)), size=MAX_CLD_SIZE, replace=False)]
        tgt_ctx.set_cld(cloud)
        cost_array = batch_tps_rpm_bij(self.src_ctx, tgt_ctx, T_init = 1e-1, T_final = 5e-3, #same as reg init and reg final?
                      outlierfrac=self.outlierfrac, outlierprior=self.outlierprior, component_cost=False)
        if self.cost_type == 'bending':
            costs = dict(zip(self.src_ctx.seg_names, cost_array))
            return costs
        else:
            raise NotImplementedError


#     def batch_cost(self, test_scene_state):
#         # TODO Dylan
#         raise NotImplementedError

class GpuTpsRpmBijRegistrationFactory(RegistrationFactory):
    # TODO Dylan
    def __init__(self, demos, filename, em_iter=1, rad_init=.1, rad_final=.005, rot_reg=np.r_[1e-4, 1e-4, 1e-1],
                    outlierprior=.1, outlierfrac=1e-2, cost_type='bending', prior_fn=None):
        super(GpuTpsRpmBijRegistrationFactory, self).__init__(demos)
        self.em_iter = em_iter
        self.rad_init = rad_init
        self.rad_final = rad_final
        self.rot_reg = rot_reg
        self.outlierprior = outlierprior
        self.outlierfrac = outlierfrac
        self.cost_type = cost_type
        self.prior_fn = prior_fn
        self.bend_coefs = np.around(loglinspace(DEFAULT_LAMBDA[0], DEFAULT_LAMBDA[1], N_ITER_CHEAP), BEND_COEF_DIGITS)
        self.exact_bend_coefs = np.around(loglinspace(EXACT_LAMBDA[0], EXACT_LAMBDA[1], N_ITER_EXACT), BEND_COEF_DIGITS)
        self.f_empty_solver = EmptySolver(MAX_CLD_SIZE, self.exact_bend_coefs)
        self.g_empty_solver = EmptySolver(MAX_CLD_SIZE, self.exact_bend_coefs)
        self.src_ctx = GPUContext(self.bend_coefs)
        self.src_ctx.read_h5(filename)

    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        """
        TODO: use em_iter (?)
        """
        if self.prior_fn is not None:
            vis_cost_xy = self.prior_fn(demo.scene_state, test_scene_state)
        else:
            vis_cost_xy = None

        old_cloud = demo.scene_state.cloud[:,:3]
        new_cloud = test_scene_state.cloud[:,:3]
        x_nd = np.array(old_cloud)
        y_md = np.array(new_cloud)
        if len(old_cloud) > MAX_CLD_SIZE:
            x_nd = x_nd[np.random.choice(range(len(x_nd)), size=MAX_CLD_SIZE, replace=False)]
        if len(new_cloud) > MAX_CLD_SIZE:
            y_md = y_md[np.random.choice(range(len(y_md)), size=MAX_CLD_SIZE, replace=False)]

        scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        scaled_y_md, targ_params = registration.unit_boxify(y_md)

        x_K_nn = tps_kernel_matrix(scaled_x_nd)
        fsolve = self.f_empty_solver.get_solver(scaled_x_nd, x_K_nn, self.exact_bend_coefs)
        y_K_nn = tps_kernel_matrix(scaled_y_md)
        gsolve = self.g_empty_solver.get_solver(scaled_y_md, y_K_nn, self.exact_bend_coefs)

        x_weights = np.ones(len(x_nd)) * 1.0/len(x_nd)
        (f,g), corr = tpsopt.registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, fsolve, gsolve,
                                                      n_iter = N_ITER_EXACT,
                                                      reg_init = EXACT_LAMBDA[0],
                                                      reg_final = EXACT_LAMBDA[1],
                                                      rad_init = self.rad_init,
                                                      rad_final = self.rad_final,
                                                      rot_reg = self.rot_reg,
                                                      outlierprior = self.outlierprior,
                                                      outlierfrac = self.outlierfrac,
                                                      vis_cost_xy = vis_cost_xy,
                                                      return_corr = True,
                                                      check_solver = False)
        bending_cost = registration.tps_reg_cost(f)
        f = registration.unscale_tps(f, src_params, targ_params)
        f._bending_cost = bending_cost # TODO: do this properly
        return Registration(demo, test_scene_state, f, corr, g=g)

#    def batch_register(self, test_scene_state):
        #given a test_scene_state, register it to all demos and return all registrations as dict (?)
        #unfortunately, batch_tps_rpm_bij doesn't provide the actual transforms

    def cost(self, demo, test_scene_state):
        raise NotImplementedError

    def batch_cost(self, test_scene_state):
        tgt_ctx = TgtContext(self.src_ctx)
        cloud = test_scene_state.cloud
        if len(cloud) > MAX_CLD_SIZE:
            cloud = cloud[np.random.choice(range(len(cloud)), size=MAX_CLD_SIZE, replace=False)]
        tgt_ctx.set_cld(cloud)
        cost_array = batch_tps_rpm_bij(self.src_ctx, tgt_ctx, T_init = 1e-1, T_final = 5e-3, #same as reg init and reg final?
                      outlierfrac=self.outlierfrac, outlierprior=self.outlierprior, component_cost=False)
        if self.cost_type == 'bending':
            costs = dict(zip(self.src_ctx.seg_names, cost_array))
            return costs
        else:
            raise NotImplementedError

class TpsRpmRegistrationFactory(RegistrationFactory):
    """
    As in:
        H. Chui and A. Rangarajan, "A new point matching algorithm for non-rigid registration," Computer Vision and Image Understanding, vol. 89, no. 2, pp. 114-141, 2003.
    """
    # TODO Dylan
    def __init__(self, demos, n_iter=N_ITER_EXACT, reg_init=EXACT_LAMBDA[0], reg_final=EXACT_LAMBDA[1],
        rad_init=.1, rad_final=.005, rot_reg=np.r_[1e-4, 1e-4, 1e-1], cost_type='bending'):
        super(TpsRpmRegistrationFactory,self).__init__(demos)
        self.n_iter = n_iter
        self.reg_init = reg_init
        self.reg_final = reg_final
        self.rad_init = rad_init
        self.rad_final = rad_final
        self.rot_reg = rot_reg
        self.cost_type = cost_type

    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        """
        TODO: use em_iter?
        """
        old_cloud = demo.scene_state.cloud
        new_cloud = test_scene_state.cloud
        x_nd = old_cloud[:,:3]
        y_md = new_cloud[:,:3]
        scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        scaled_y_md, targ_params = registration.unit_boxify(y_md)
        f, corr = registration.tps_rpm(scaled_x_nd, scaled_y_md,
                                    n_iter = self.n_iter,
                                    reg_init = self.reg_init,
                                    reg_final = self.reg_final,
                                    rad_init = self.rad_init,
                                    rad_final = self.rad_final,
                                    rot_reg = self.rot_reg,
                                    return_corr = True,
                                    plotting = plotting,
                                    plot_cb = plot_cb)
        bending_cost = registration.tps_reg_cost(f)
        f = registration.unscale_tps(f, src_params, targ_params)
        f._bending_cost = bending_cost # TODO: do this properly
        return Registration(demo, test_scene_state, f, corr)

    # def batch_register(self, test_scene_state):
    #     raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        raise NotImplementedError #same as others? should this be higher-level?
    
    # def batch_cost(self, test_scene_state):
    #     raise NotImplementedError

class GpuTpsRpmRegistrationFactory(RegistrationFactory):
    # TODO Dylan
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

class TpsSegmentRegistrationFactory(RegistrationFactory):
    def __init__(self, demos):
        # TODO
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        # TODO
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError

class TpsnRpmRegistrationFactory(RegistrationFactory):
    """
    TPS-RPM using normals information
    """
    def __init__(self, demos):
        # TODO
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        # TODO
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError
