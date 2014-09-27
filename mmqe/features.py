"""
Functions and classes for computing features
"""
import h5py

import numpy as np

from tpsopt.batchtps import SrcContext, TgtContext, batch_tps_rpm_bij, GPUContext
from tpsopt.registration import unit_boxify

import IPython as ipy
from pdb import pm, set_trace


class Feature(object):
    """
    base class for computing features
    """
    def __init__(self, actionfile):
        raise NotImplementedError

    def feature(self, state, **kwargs):
        """
        returns the feature for this state/segname
        """
        feats = self.features(state, **kwargs)
        ind = self.get_ind(segname)
        return feats[ind]

    def features(self, state, **kwargs):
        """
        returns a dictionary mapping segnames to features
        """
        raise NotImplementedError

    def select_best(self, state, k = 1):
        scores = zip(self.src_ctx.seg_names, np.dot(self.features(state), self.weights))
        scores = sorted(scores, key=lambda x: -x[1]) # we want to max
        return ([x[0] for x in scores[:k]], [x[1] for x in scores[:k]])

    def load_weights(self, fname):
        f = h5py.File(fname, 'r')
        weights = f['weights'][:]
        w0 = f['w0'][()]
        f.close()
        assert weights.shape == self.weights.shape
        self.weights = weights
        self.w0 = w0

    def get_ind(self, a):
        raise NotImplementedError

    def get_timestep(self, segname):
        return int(re.search("seg(?P<demo_timestep>\d+)",
                             segname).group('demo_timestep'))

class BatchRCFeats(Feature):

    def __init__(self, actionfile, action_list=[]):
        self.src_ctx = SrcContext()
        self.src_ctx.read_h5(actionfile)
        self.tgt_cld = None
        self.tgt_ctx = TgtContext(self.src_ctx)
        self.name2ind = dict([(s, i) for i, s in enumerate(self.src_ctx.seg_names)])
        self.costs = np.zeros(self.src_ctx.N)
        self.N = len(self.src_ctx.seg_names)
        self.indicators = np.eye(self.N)
        self.weights = np.r_[-1, np.zeros(self.N)]
        self.w0 = 0

    def features(self, state, **kwargs):
        self.tgt_cld = state.cloud
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx)
        return np.c_[self.costs, self.indicators]

    def set_name2ind(self, action_list):
        self.name2ind = dict([(s, i) for i, s in enumerate(action_list)])

    def get_ind(self, a):
        return self.name2ind[a]

    @staticmethod
    def get_size(num_actions):
        return num_actions + 1


class MulFeats(BatchRCFeats):

    N_costs = 5

    def __init__(self, actionfile):
        BatchRCFeats.__init__(self, actionfile)
        x = np.array([-1 for _ in range(MulFeats.N_costs)])
        self.weights = np.r_[x, np.zeros(self.N)]

    def features(self, state, **kwargs):
        self.tgt_cld = state.cloud
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx, component_cost=True)
        return np.c_[self.costs, self.indicators]

    def get_ind(self, a):
        return self.name2ind[a]

    @staticmethod
    def get_size(num_actions):
        return BatchRCFeats.get_size(num_actions) + MulFeats.N_costs - 1

class SimpleMulFeats(MulFeats):
    
    N_costs = 3
    def __init__(self, actionfile):
        BatchRCFeats.__init__(self, actionfile)
        x = np.array([-1 for _ in range(SimpleMulFeats.N_costs)])
        self.weights = np.r_[x, np.zeros(self.N)]
 
    def features(self, state, **kwargs):
        self.tgt_cld = state.cloud
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx, component_cost=True)[:, :SimpleMulFeats.N_costs]
        return np.c_[self.costs, self.indicators]

    @staticmethod
    def get_size(num_actions):
        return BatchRCFeats.get_size(num_actions) +SimpleMulFeats.N_costs - 1

def get_quad_terms(vec):
    N = vec.shape[0]
    v_t_v = np.dot(vec[:, None], vec[None, :])
    inds = np.triu_indices(N)
    return np.r_[vec, v_t_v[inds]]

class LandmarkFeats(MulFeats):
    
    def __init__(self, actionfile):
        MulFeats.__init__(self, actionfile)
        self.landmark_ctx = None

    def set_landmark_file(self, landmarkf):
        self.landmark_ctx = GPUContext()
        self.landmark_ctx.read_h5(landmarkf)
        self.landmark_targ_ctx = TgtContext(self.landmark_ctx)
        self.weights = np.zeros(self.src_ctx.N + self.landmark_ctx.N + MulFeats.N_costs)

    def features(self, state, **kwargs):
        mul_feats = MulFeats.features(self, state)
        self.landmark_targ_ctx.set_cld(state.cloud)
        landmark_feats = batch_tps_rpm_bij(self.landmark_ctx, self.landmark_targ_ctx)
        landmark_feats = np.exp(-landmark_feats)
        landmark_feats /= np.sum(landmark_feats)
        self.costs = np.c_[mul_feats, np.tile(landmark_feats, (self.src_ctx.N, 1))]
        return self.costs

    @staticmethod
    def get_size(num_actions, num_landmarks=70):
        return num_actions + num_landmarks + MulFeats.N_costs
    
    

class QuadMulFeats(BatchRCFeats):         

    N_feats = sum([x+1 for x in range(MulFeats.N_costs)]) + MulFeats.N_costs

    def __init__(self, actionfile):
        BatchRCFeats.__init__(self, actionfile)
        self.weights = np.zeros(QuadMulFeats.get_size(self.N))

    def features(self, state, **kwargs):
        self.tgt_cld = state.cloud
        self.tgt_ctx.set_cld(self.tgt_cld)
        costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx, component_cost=True)
        self.costs = np.zeros((self.N, QuadMulFeats.N_feats))
        for i in range(self.N):
            self.costs[i, :] = get_quad_terms(costs[i])
        return np.c_[self.costs, self.indicators]

    def get_ind(self, a):
        return self.name2ind[a]

    @staticmethod
    def get_size(num_actions):
        return num_actions + QuadMulFeats.N_feats

class TimestepFeats(BatchRCFeats):

    N_timesteps = 7
    def __init__(self, actionfile):
        BatchRCFeats.__init__(self, actionfile)
        self.weights = np.zeros(TimestepFeats.get_size(self.N))
        self.ts_indicators = np.zeros([self.N, TimestepFeats.N_timesteps])

    def features(self, state, timestep=-1, **kwargs):
        if timestep == -1:
            raise ValueError('timestep must be input')
        if timestep < TimestepFeats.N_timesteps:
            self.ts_indicators[:, timestep] = 1
        return np.c_[self.ts_indicators, self.indicators]

    @staticmethod
    def get_size(num_actions):
        return num_actions + TimestepFeats.N_timesteps


class TimestepActionFeats(TimestepFeats):

    def __init__(self, actionfile):
        TimestepFeats.__init__(self, actionfile)
        self.n_quad = self.N * TimestepFeats.N_timesteps
        self.quad_feats = np.zeros([self.N, self.n_quad])
        self.weights = np.zeros(TimestepActionFeats.get_size(self.N))

    def features(self, state, timestep=-1, **kwargs):
        TimestepFeats.features(self, state, timestep)
        for i in range(self.N):
            self.quad_feats[i,:] = np.reshape(np.dot(self.indicators[i][:,None], self.ts_indicators[0][None,:]), self.n_quad)
        return np.c_[self.ts_indicators, self.indicators, self.quad_feats]

    @staticmethod
    def get_size(num_actions):
        return num_actions + TimestepFeats.N_timesteps*(num_actions+1)

class TimestepActionMulFeats(TimestepFeats, SimpleMulFeats):

    def __init__(self, actionfile):
        TimestepFeats.__init__(self, actionfile)
        self.n_quad = self.N * TimestepFeats.N_timesteps
        self.quad_feats = np.zeros([self.N, self.n_quad])
        self.weights = np.zeros(TimestepActionMulFeats.get_size(self.N))

    def features(self, state, timestep=-1, **kwargs):
        TimestepFeats.features(self, state, timestep)
        SimpleMulFeats.features(self, state)
        for i in range(self.N):
            self.quad_feats[i,:] = np.reshape(np.dot(self.indicators[i][:,None], self.ts_indicators[0][None,:]), self.n_quad)
        return np.c_[self.costs, self.ts_indicators, self.indicators, self.quad_feats]

    @staticmethod
    def get_size(num_actions):
        return SimpleMulFeats.get_size(num_actions) + TimestepFeats.N_timesteps*(num_actions+1)

