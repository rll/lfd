from ropesimulation.constants import ROPE_RADIUS, DS_SIZE
from tpsopt.constants import EXACT_LAMBDA, N_ITER_EXACT
import h5py
from rapprentice import registration, tps_registration, ropesim
import numpy as np
import IPython as ipy

class Transfer(object):
    def __init__(self, args_eval):
        self.args_eval = args_eval
    
    def __getstate__(self):
        transfer = Transfer(self.args_eval)
        return transfer.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        #TODO check if instance is already initialized
    
    def initialize(self):
        self.actions = h5py.File(self.args_eval.actionfile, 'r')

    def get_scaled_action_cloud(self, state, action):
        ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
        ds_g = self.actions[action]['inv'][ds_key]
        scaled_x_na = ds_g['scaled_cloud_xyz'][:]
        src_params = (ds_g['scaling'][()], ds_g['scaled_translation'][:])
        return scaled_x_na, src_params

    def get_action_cloud(self, action, args_eval):
        rope_nodes = self.get_action_rope_nodes(action, args_eval)
        cloud = ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample_rad=args_eval.upsample_rad)
        return cloud
    
    def get_action_cloud_ds(self, action, args_eval):
        if args_eval.downsample:
            ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
            return self.actions[action]['inv'][ds_key]['cloud_xyz']
        else:
            return self.get_action_cloud(action, args_eval)
    
    def get_action_rope_nodes(self, action, args_eval):
        rope_nodes = self.actions[action]['cloud_xyz'][()]
        return ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample=args_eval.upsample)
    
    def register_tps(self, state, action, args_eval, reg_type='bij'):
        scaled_x_nd, src_params = self.get_scaled_action_cloud(state, action)
        new_cloud = state.cloud
        if reg_type == 'bij':
            vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
            y_md = new_cloud[:,:3]
            scaled_y_md, targ_params = registration.unit_boxify(y_md)
            rot_reg = np.r_[1e-4, 1e-4, 1e-1]
            reg_init = EXACT_LAMBDA[0]
            reg_final = EXACT_LAMBDA[1]
            n_iter = N_ITER_EXACT
            (f,g), corr = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=rot_reg, n_iter=n_iter, 
                                      reg_init=reg_init, reg_final=reg_final, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy, 
                                      return_corr=True)
            f = registration.unscale_tps(f, src_params, targ_params)
            f._bend_coef = reg_final
            f._rot_coef = rot_reg
            f._wt_n = corr.sum(axis=1)
        else:
            raise RuntimeError('invalid registration type')
        return f, corr
