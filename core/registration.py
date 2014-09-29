from __future__ import division

from constants import EXACT_LAMBDA, DEFAULT_LAMBDA, N_ITER_EXACT, N_ITER_CHEAP
import numpy as np
from rapprentice import registration
from tn_rapprentice import registration as tn_registration
from tn_rapprentice import tps_n_rpm as tps_n_rpm
from tn_eval import tps_utils
from tn_rapprentice import krig_utils as ku

import tpsopt
from constants import BEND_COEF_DIGITS, MAX_CLD_SIZE
from tpsopt.tps import tps_kernel_matrix
from tpsopt.registration import loglinspace
from tpsopt.batchtps import GPUContext, TgtContext, SrcContext, batch_tps_rpm_bij
from tpsopt.transformations import EmptySolver, TPSSolver

import openravepy

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
    def __init__(self, demos, n_iter=N_ITER_EXACT, em_iter=1, reg_init=1, 
        reg_final=1e-4, rad_init=.01, rad_final=.001, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
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
        f = registration.fit_ThinPlateSpline(x_nd,corr.dot(y_md),bend_coef=self.bend_coef)
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
        #   

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
        rad_init=.05, rad_final=.01, rot_reg=np.r_[1e-4, 1e-4, 1e-1], cost_type='bending'):
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
        f, corr = tn_registration.tps_rpm(scaled_x_nd, scaled_y_md,
                                    n_iter = self.n_iter,
                                    reg_init = self.reg_init,
                                    reg_final = self.reg_final,
                                    rad_init = self.rad_init,
                                    rad_final = self.rad_final,
                                    rot_reg = self.rot_reg,
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
    #def __init__(self, demos, cost_type="bending", prior_fn=None, temp_init=.005, temp_final=.0001, 
    #    bend_init=1e2, bend_final=1e-3, outlierfrac=1e-2, outlierprior=1e-2, normal_coef_init=1e-8, normal_coef_final=3e-4, normal_temp_init=.5,normal_temp_final=.02, sim=None):
    #def __init__(self, demos, cost_type="bending", prior_fn=None, temp_init=.005, temp_final=.0001, 
    #    bend_init=1e2, bend_final=5e-4, outlierfrac=1e-2, outlierprior=1e-2, normal_coef_init=1e-15, normal_coef_final=1e-4, normal_temp_init=.4,normal_temp_final=.01, sim=None):
    #def __init__(self, demos, cost_type="bending", prior_fn=None, temp_init=.005, temp_final=.00008, 
    #            bend_init=1e4, bend_final=8e-4, outlierfrac=1e-3, outlierprior=1e-3, normal_coef_init=1e-15, normal_coef_final=1e-4, normal_temp_init=1,normal_temp_final=.01, sim=None):
    

    def __init__(self, demos, cost_type="bending", prior_fn=None, temp_init=.005, temp_final=.0002, 
                bend_init=1e4, bend_final=.1, outlierfrac=1e-3, outlierprior=1e-3, normal_coef_init=1e-8, normal_coef_final=0.001, normal_temp_init=.1,normal_temp_final=.1, sim=None):
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
        self.normal_coef_init = normal_coef_init
        self.normal_coef_final = normal_coef_final
        self.normal_temp_init = normal_temp_init
        self.normal_temp_final = normal_temp_final
        self.sim = sim
        self.i0,self.i1=0,0

    def register(self, demo, test_scene_state, plotting=True, plot_cb=None):
        # TODO
        """
        TODO: use em_iter
        """
        p=False
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
        """
        Epts = np.r_[x_nd[0:8],x_nd[0:8],x_nd[0:8]]
        Exs = np.r_[np.tile(np.array([[-1,0,0],[-1,0,0],[1,0,0],[1,0,0]]), (2,1))]
        Exs = np.r_[Exs, np.tile(np.array([[0,-1,0],[0,1,0]]),(4,1))]
        Exs = np.r_[Exs,np.tile(np.array([0,0,1]), (8,1))]
        Eys = Exs
        """

        
        from rapprentice import berkeley_pr2
        T_w_k = berkeley_pr2.get_kinect_transform(self.sim.robot)
        T_z = np.eye(4)
        T_z[:3,:3] = openravepy.rotationMatrixFromAxisAngle(np.r_[0,0,np.pi])
        T_z[0,3] = 1.25

        """
        if x_nd.shape[0]>y_md.shape[0]:
            x_nd = x_nd[:y_md.shape[0],]
        else:
            y_md = y_md[:x_nd.shape[0],]
            """
        Epts = x_nd
        #ipy.embed()
        Exs = tps_utils.find_all_normals_naive(x_nd[0:self.i0,],orig_cloud = demo.scene_state.full_cloud,wsize=0.02,flip_away=True, origin=T_w_k[:3,3])
        Exs = np.r_[Exs,tps_utils.find_all_normals_naive(x_nd[self.i0:,],orig_cloud = demo.scene_state.full_cloud,wsize=0.02,flip_away=True, origin=T_z[:3,3])]
        #test_scene_state.full_cloud
        Eys = tps_utils.find_all_normals_naive(y_md[0:self.i1,],orig_cloud = test_scene_state.full_cloud,wsize=0.02,flip_away=True, origin=T_w_k[:3,3])
        Eys = np.r_[Eys,tps_utils.find_all_normals_naive(y_md[self.i1:,],orig_cloud = test_scene_state.full_cloud,wsize=0.02,flip_away=True, origin=T_z[:3,3])]
        #Eys=Exs
        handles = []
        for i in range(x_nd.shape[0]):
            pass#handles.append(self.sim.env.drawlinestrip(np.array([Epts[i],np.array(Epts[i]+Exs[i]/10)]),5,(0,1,0,1)))
        self.sim.viewer.Step()
        for i in range(y_md.shape[0]):
            pass#handles.append(self.sim.env.drawlinestrip(np.array([y_md[i],np.array(y_md[i]+Eys[i]/10)]),5,(1,0,0,1)))
        self.sim.viewer.Step()
        #self.sim.viewer.Idle()
        #ipy.embed()

        #ipy.embed()
        """
        Epts = np.r_[x_nd[0:8],x_nd[0:8],x_nd[0:8]]
        Exs = np.r_[np.tile(np.array([[-1,0,0],[-1,0,0],[1,0,0],[1,0,0]]), (2,1))]
        Exs = np.r_[Exs, np.tile(np.array([[0,-1,0],[0,1,0]]),(4,1))]
        Exs = np.r_[Exs,np.tile(np.array([0,0,1]), (16,1))]
        Eys = Exs
        """
        """
        ipy.embed()
        z = open("inputs.txt",'w')
        np.set_printoptions(threshold=np.nan)
        z.write("x_nd=np."+repr(x_nd)+"\n\n")
        z.write("y_md=np."+repr(y_mds)+"\n\n")
        z.write("Epts=np."+repr(Epts)+"\n\n")
        z.write("Exs=np."+repr(Exs)+"\n\n")
        z.write("Eys=np."+repr(Eys)+"\n\n")
        """
        #f,corr1,_ = tn_registration.tps_n_rpm_final_hopefully(x_nd, y_md, Exs, Eys, temp_init=self.temp_init,  temp_final=self.temp_final, bend_init=self.bend_init, bend_final=self.bend_final,
        #            outlierfrac = self.outlierfrac, outlierprior = self.outlierprior, normal_coef = self.normal_coef_init, normal_temp = .05)
        #temp_init=.5,  temp_final=.001, bend_init=1e-1, bend_final=1e-3,
        #             wsize=.1, normal_coef = 1e-1,  normal_temp = 1e2, beta=0

        f,corr1,corr_nm_edge = tn_registration.tps_n_rpm_final_hopefully(x_nd, y_md, Exs, Eys, Epts, temp_init=self.temp_init,  temp_final=self.temp_final, bend_init=self.bend_init, bend_final=self.bend_final,
                     normal_coef_init = self.normal_coef_init, n_iter=20, normal_coef_final = self.normal_coef_final, normal_temp_init = self.normal_temp_init, normal_temp_final = self.normal_temp_final, beta=0, sim=self.sim, plotting=p,
                     outlierprior=self.outlierprior,outlierfrac=self.outlierfrac)
        #ipy.embed()
        ##f = tn_registration.fit_ThinPlateSpline(x_nd,corr1.dot(y_md),bend_coef=1)
        #ipy.embed()
        #ipy.embed()
        print np.sum(corr1),np.sum(corr_nm_edge)
        #ipy.embed()
        #f = registration.fit_ThinPlateSpline(x_nd,corr1.dot(y_md),bend_coef=1)
        #ipy.embed()
        #f,corr1,_ = tn_registration.tps_rpm_double_corr(x_nd, y_md, Exs, Eys, temp_init=self.temp_init,  temp_final=self.temp_final, bend_init=self.bend_init, bend_final=self.bend_final,
        #            outlierfrac = self.outlierfrac, outlierprior = self.outlierprior, normal_coef_init = self.normal_coef_init, normal_coef_final = self.normal_coef_init)
        
        #ipy.embed()
        #a,b,c = ku.krig_fit1Normal1(1.5, x_nd, y_md, Epts, Exs, Eys, bend_coef = 1e-9, normal_coef = 1e7, wt_n = None, rot_coefs = 1e-5)
        #f = tn_registration.fit_KrigingSpline(x_nd, Epts, Exs, corr1.dot(y_md), corr_nm_edge.dot(Eys), bend_coef = 1,normal_coef = 1)
        #bend_coef=1e-13,normal_coef=1e7
        #ipy.embed()
        #f = tn_registration.fit_KrigingSpline(x_nd, Epts, Exs, x_nd, Exs, bend_coef = 1e2,normal_coef = 1e2, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)
        #f1 = tn_registration.fit_KrigingSpline(x_nd.copy(), Epts.copy(), Exs.copy(), y_md.copy(), Eys.copy(), bend_coef = 1e-9,normal_coef = 1e7, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)
        
        #f = tn_registration.fit_KrigingSpline(x_nd.copy(), Epts, Exs, y_md, Eys, bend_coef = 1e-9,normal_coef = 1e7, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)
        #y_md = corr.dot(y_md)
        #f = registration.fit_ThinPlateSpline(x_nd, y_md, bend_coef=self.bend_coef)
        #f = tn_registration.fit_KrigingSpline(x_nd, Epts, Exs, y_md, Eys, bend_coef = self.bend_coef,normal_coef = self.normal_coef, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)

        #bending_cost = registration.tps_reg_cost(f)
        f_tmp = f
        #f = registration.unscale_tps(f, src_params, targ_params)
        #f._old = f_tmp
        f._bending_cost = 0 # TODO: do this properly

        return Registration(demo, test_scene_state, f, None)
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError

class TpsnRegistrationFactory(RegistrationFactory):
    """
    TPS using normals information
    """
    def __init__(self, demos, bend_coef=1e-9, normal_coef=1e7, prior_fn=None, cost_type='bending', sim=None):
        """
        TODO: do something better for default parameters and write comment
        """
        super(TpsnRegistrationFactory, self).__init__(demos)
        self.cost_type = cost_type
        self.bend_coef = bend_coef
        self.prior_fn = prior_fn
        self.normal_coef = normal_coef
        self.sim = sim
    
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
        
        from rapprentice import berkeley_pr2
        T_w_k = berkeley_pr2.get_kinect_transform(self.sim.robot)

        if x_nd.shape[0]>y_md.shape[0]:
            x_nd = x_nd[:y_md.shape[0],]
        else:
            y_md = y_md[:x_nd.shape[0],]

        Epts = x_nd
        Exs = tps_utils.find_all_normals_naive(x_nd, orig_cloud = demo.scene_state.full_cloud,wsize=0.02,flip_away=True, origin=T_w_k[:3,3])
        #test_scene_state.full_cloud
        Eys = Exs

        Epts = np.r_[x_nd[0:8],x_nd[0:8],x_nd[0:8]]
        Exs = np.r_[np.tile(np.array([[-1,0,0],[-1,0,0],[1,0,0],[1,0,0]]), (2,1))]
        Exs = np.r_[Exs, np.tile(np.array([[0,-1,0],[0,1,0]]),(4,1))]
        Exs = np.r_[Exs,np.tile(np.array([0,0,1]), (8,1))]
        Eys = Exs

        
        handles = []
        for i in range(Epts.shape[0]):
            handles.append(self.sim.env.drawlinestrip(np.array([Epts[i],np.array(Epts[i]+Exs[i]/20)]),2,(0,1,0,1)))
        self.sim.viewer.Step()
        #self.sim.viewer.Idle()

        #f = tn_registration.fit_KrigingSpline(x_nd.copy(), Epts.copy(), Exs.copy(), y_md.copy(), Eys.copy(), bend_coef = self.bend_coef, normal_coef = self.normal_coef, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)
        f = tn_registration.fit_KrigingSpline(x_nd.copy(), Epts.copy(), Exs.copy(), y_md.copy(), Eys.copy(), bend_coef = self.bend_coef, normal_coef = self.normal_coef, wt_n=None, alpha = 1.5, rot_coefs = 1e-5)
        f_tmp = f


        #f._old = f_tmp
        f._bending_cost = 0 # TODO: do this properly
        return Registration(demo, test_scene_state, f, None)
    
    def cost(self, demo, test_scene_state):
        # TODO
        raise NotImplementedError