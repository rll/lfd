from __future__ import division

from constants import EXACT_LAMBDA, N_ITER_EXACT
import numpy as np
from rapprentice import registration

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
        J. Schulman, J. Ho, C. Lee, and P. Abbeel, "Learning from Demonstrations through the Use of Non-Rigid Registration," in Proceedings of the 16th International Symposium on Robotics Research (ISRR), 2013.
    """
    def __init__(self, demos, n_iter=N_ITER_EXACT, reg_init=EXACT_LAMBDA[0], reg_final=EXACT_LAMBDA[1], rad_init=.1, rad_final=.005, rot_reg=np.r_[1e-4, 1e-4, 1e-1], outlierprior=.1, outlierfrac=1e-2, cost_type='bending', prior_fn=None):
        """
        TODO: do something better for default parameters and write comment
        """
        super(TpsRpmBijRegistrationFactory, self).__init__(demos)
        self.n_iter = n_iter
        self.reg_init = reg_init
        self.reg_final = reg_final
        self.rad_init = rad_init
        self.rad_final = rad_final
        self.rot_reg = rot_reg
        self.outlierprior = outlierprior
        self.outlierfrac = outlierfrac
        self.cost_type = cost_type
        self.prior_fn = prior_fn
        
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
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

    def batch_register(self, test_scene_state):
        # TODO Dylan
        raise NotImplementedError

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

class TpsRpmRegistrationFactory(RegistrationFactory):
    """
    As in:
        H. Chui and A. Rangarajan, "A new point matching algorithm for non-rigid registration," Computer Vision and Image Understanding, vol. 89, no. 2, pp. 114-141, 2003.
    """
    def __init__(self, demos):
        # TODO
        raise NotImplementedError
    
    def register(self, demo, test_scene_state, plotting=False, plot_cb=None):
        # TODO
        raise NotImplementedError
    
    def batch_register(self, test_scene_state):
        # TODO Dylan
        raise NotImplementedError
    
    def cost(self, demo, test_scene_state):
        # TODO Dylan
        raise NotImplementedError
    
    def batch_cost(self, test_scene_state):
        # TODO Dylan
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
