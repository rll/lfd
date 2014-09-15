from __future__ import division

from constants import EXACT_LAMBDA, DEFAULT_LAMBDA, N_ITER_EXACT, N_ITER_CHEAP
import numpy as np
from rapprentice import registration
from tn_rapprentice import registration as tn_registration
from tn_rapprentice import tps_n_rpm as tps_n_rpm
from tn_eval import tps_utils

import tpsopt
from constants import BEND_COEF_DIGITS, MAX_CLD_SIZE
from tpsopt.tps import tps_kernel_matrix
from tpsopt.registration import loglinspace
from tpsopt.batchtps import GPUContext, TgtContext, SrcContext, batch_tps_rpm_bij
from tpsopt.transformations import EmptySolver, TPSSolver

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
    def __init__(self, demos, n_iter=N_ITER_EXACT, em_iter=1, reg_init=EXACT_LAMBDA[0], 
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
        self.bend_coef = 0
        self.normal_coef= 0
        
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
        #scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        #scaled_y_md, targ_params = registration.unit_boxify(y_md)
        x_weights = np.ones(len(old_cloud)) * 1.0/len(old_cloud)
        (f,g), corr = registration.tps_rpm_bij(x_nd,y_md,
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

        #bending_cost = registration.tps_reg_cost(f)
        f_tmp = f
        #f = registration.unscale_tps(f, src_params, targ_params)
        f._old = f_tmp
        f._bending_cost = 0 # TODO: do this properly
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
            #x_nd = old_cloud[np.random.random_integers(len(old_cloud)-1, size=min(MAX_CLD_SIZE, len(old_cloud)))]
        if len(new_cloud) > MAX_CLD_SIZE:
            y_md = y_md[np.random.choice(range(len(y_md)), size=MAX_CLD_SIZE, replace=False)]
            #y_md = new_cloud[np.random.random_integers(len(new_cloud)-1, size=min(MAX_CLD_SIZE, len(new_cloud)))]

        # if len(x_nd) != len(old_cloud) or len(y_md) != len(new_cloud):
        #     ipy.embed()

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
    def __init__(self, demos, cost_type="bending", prior_fn=None, temp_init=.1, temp_final=.01, bend_init=.01, bend_final=.001, outlierfrac=1e-2, outlierprior=.1, normal_weight_init=5, normal_weight_final=.5, normal_coef_init=1, normal_coef_final=.5):
        """
        TODO: do something better for default parameters and write comment
        """
        super(TpsnRpmRegistrationFactory, self).__init__(demos)
        self.cost_type = cost_type
        self.prior_fn = prior_fn
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.bend_init = bend_init
        self.bend_final = bend_final
        self.outlierfrac = outlierfrac
        self.outlierprior = outlierprior
        self.normal_weight_init = normal_weight_init
        self.normal_weight_final = normal_weight_final
        self.normal_coef_init = normal_coef_init
        self.normal_coef_final = normal_coef_final

    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        # TODO
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
        #scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        #scaled_y_md, targ_params = registration.unit_boxify(y_md)

        #f = registration.ThinPlateSpline(3)
        #Epts = np.r_[x_nd[0:8],x_nd[0:8]]
        #Exs = tps_utils.find_all_normals_naive(x_nd, wsize = 0.05)
        #Exs = np.r_[np.tile(np.array([-1,0,0]),(2,1)),np.tile(np.array([-1,0,0]),(2,1)),np.tile(np.array([-1,0,0]),(2,1)),np.tile(np.array([-1,0,0]),(2,1)),np.tile(np.array([0,0,1]),(8,1))]
        #Exs = np.r_[Exs,Exs]
        #Eys = Exs

        Epts = np.array([[.45,-.2,1], [.55,0,1]])
        Epts = np.r_[Epts,x_nd[0:8]]
        Exs = np.tile(np.array([0,0,1]), (10,1))
        Eys = Exs

        f,corr1,_ = tps_n_rpm.tps_rpm_double_corr(x_nd, y_md, Epts, Exs, Eys, temp_init=self.temp_init,  temp_final=self.temp_final, bend_init=self.bend_init, bend_final=self.bend_final,
                    outlierfrac = self.outlierfrac, outlierprior = self.outlierprior, normal_weight_init = self.normal_weight_init, normal_weight_final = self.normal_weight_final, normal_coef_init = self.normal_coef_init, normal_coef_final = self.normal_coef_final)
        
        #y_md = corr.dot(y_md)
        #f = registration.fit_ThinPlateSpline(x_nd, y_md, bend_coef=self.bend_coef)
        #f = tn_registration.fit_KrigingSpline(x_nd, Epts, Exs, y_md, Eys, bend_coef = self.bend_coef,normal_coef = self.normal_coef, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)

        #bending_cost = registration.tps_reg_cost(f)
        f_tmp = f
        #f = registration.unscale_tps(f, src_params, targ_params)
        f._old = f_tmp
        f._bending_cost = 0 # TODO: do this properly
        return Registration(demo, test_scene_state, f, corr1)
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError

class TpsnRegistrationFactory(RegistrationFactory):
    """
    TPS using normals information
    """
    def __init__(self, demos, bend_coef=1e-9, normal_coef=1e-9, prior_fn=None, cost_type='bending'):
        """
        TODO: do something better for default parameters and write comment
        """
        super(TpsnRegistrationFactory, self).__init__(demos)
        self.cost_type = cost_type
        self.bend_coef = bend_coef
        self.prior_fn = prior_fn
        self.normal_coef = normal_coef
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        # TODO
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

        Epts = np.array([[.45,-.2,1], [.55,0,1]])
        Epts = np.r_[Epts,x_nd[0:8]]
        Exs = np.tile(np.array([0,0,1]), (10,1))
        Eys = Exs

        f = tn_registration.fit_KrigingSpline(x_nd, Epts, Exs, y_md, Eys, bend_coef = self.bend_coef,normal_coef = self.normal_coef, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)

        f_tmp = f

        f._old = f_tmp
        f._bending_cost = 0 # TODO: do this properly
        return Registration(demo, test_scene_state, f, None)
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError