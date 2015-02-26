"""
Functions and classes to compute constraints for the optimizations in max_margin.py
"""

import numpy as np
import sys, h5py

import ipdb

from features import BatchRCFeats

class ConstraintGenerator(object):
    """
    base class for computing MM constraints
    """
    def __init__(self, feature, margin, actionfile):
        self.actionfile = h5py.File(actionfile, 'r')
        self.feature = feature
        self.margin  = margin
        self.n_constrs = 0

    def compute_constrs(self, state, exp_a, timestep=-1):
        """
        computes a single constraint
        None for the expert action indicates this is an unrecoverable state
        """
        #ipdb.set_trace()
        phi              = self.feature.features(state, timestep=timestep)
        if exp_a == 'failure':
            return -1, phi, -1
        feat_i           = self.feature.get_ind(exp_a)
        f_mask           = np.ones(self.feature.N, dtype=np.bool)
        f_mask[feat_i]   = 0
        margin_i         = self.margin.get_ind(exp_a)        
        m_mask           = np.ones(self.feature.N, dtype=np.bool)
        m_mask[margin_i] = 0
        margins          = self.margin.get_margins(state, exp_a)

        exp_phi = phi[feat_i]        
        return exp_phi, phi[f_mask, :], margins[m_mask]

    def store_constrs(self, exp_phi, phi, margins, exp_a, outfile, reward=None, constr_k=None):
        if constr_k is None:
            constr_k = str(self.n_constrs)
        self.n_constrs += 1
        constr_g = outfile.create_group(constr_k)
        constr_g['exp_action_name'] = exp_a
        constr_g['exp_phi'] = exp_phi
        constr_g['rhs_phi'] = phi
        constr_g['margin'] = margins
        if reward is not None:
            constr_g['reward'] = reward


class Margin(object):
    """
    base clase for computing margins
    """
    
    def __init__(self, actions):
        self.actions = actions
        self.N = len(actions)

    def get_margins(self, state, a):
        return np.ones(self.N)

    def get_ind(self, s):
        raise NotImplementedError

class BatchCPMargin(Margin):
    
    def __init__(self, feature):
        Margin.__init__(self, feature.src_ctx.seg_names)
        self.feature = feature
        self.src_ctx = feature.src_ctx
        self.tgt_ctx = feature.tgt_ctx
        self.name2ind = feature.name2ind

    def get_margins(self, state, a):
        """
        returns a margin that is the normalized sum of distances of warped trajectories
        to the warped trajectory associated with a

        assumes that self.feature has already been run through batch_tps_rpm_bij
        with the appropriate target (i.e. the state this is called with)
        """
        return self.src_ctx.traj_cost(a, self.tgt_ctx)

    def get_ind(self, s):
        return self.feature.get_ind(s)

