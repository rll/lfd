from __future__ import division

import h5py
import numpy as np
import scipy.linalg
from tpsopt import precompute
import tps
import hashlib

class TpsSolver(object):
    """
    Fits thin plate spline to data using precomputed matrix products
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef):
        for b in bend_coefs:
            if b not in NON:
                raise RuntimeError("No precomputed NON for bending coefficient {}".format(b))
        self.rot_coef = rot_coef
        self.n, self.d  = x_nd.shape
        self.bend_coefs = bend_coefs
        self.N          = N
        self.QN         = QN        
        self.NON        = NON
        self.NR         = NR
        self.x_nd       = x_nd
        self.K_nn       = K_nn
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        if bend_coef not in self.bend_coefs:
            raise RuntimeError("No precomputed NON for bending coefficient {}".format(bend_coef))
        WQN = wt_n[:, None] * self.QN
        lhs = self.NON[bend_coef] + self.QN.T.dot(WQN)
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = scipy.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class TpsSolverFactory(object):
    def __init__(self, cache_fname=None):
        """
        cache_fname: h5 file with 'solver' key to load/save precomputed matrix products. if None, no caching is done.
        """
        if cache_fname is not None:
            self.file = h5py.File(cache_fname)
        else:
            self.file = None
    
    def _get_solver(self, x_nd, bend_coefs, rot_coef):
        K_nn = tps.tps_kernel_matrix(x_nd)
        N, QN, NON, NR = precompute.get_exact_solver(x_nd, K_nn, bend_coefs, rot_coef)
        return N, QN, NON, NR, K_nn
    
    def get_solver(self, x_nd, bend_coefs, rot_coef):
        if self.file is None:
            N, QN, NON, NR, K_nn = self._get_solver(x_nd, bend_coefs, rot_coef)
        else:
            sha1 = hashlib.sha1(np.ascontiguousarray(x_nd))
            sha1.update(rot_coef)
            x_rot_hash = sha1.hexdigest()
            if 'solver' in self.file:
                solver_g = self.file['solver']
            else:
                solver_g = self.file.create_group('solver')
            if x_rot_hash in solver_g and \
            np.all(x_nd == solver_g[x_rot_hash]['x_nd'][:]) and \
            np.all(rot_coef == solver_g[x_rot_hash]['rot_coef'][:]): # no hash collision
                x_rot_g = solver_g[x_rot_hash]
                N = x_rot_g['N'][:]
                QN = x_rot_g['QN'][:]
                NON_g = x_rot_g['NON']
                NON = {}
                for b in NON_g.keys():
                    NON[float(b)] = NON_g[b][:]
                NR = x_rot_g['NR'][:]
                K_nn = x_rot_g['K_nn'][:]
            else:
                N, QN, NON, NR, K_nn = self._get_solver(x_nd, bend_coefs, rot_coef)
                if x_rot_hash in solver_g:
                    del solver_g[x_rot_hash]
                x_rot_g = solver_g.create_group(x_rot_hash)
                x_rot_g['x_nd'] = x_nd
                x_rot_g['rot_coef'] = rot_coef
                x_rot_g['N'] = N
                x_rot_g['QN'] = QN
                NON_g = x_rot_g.create_group('NON')
                for b, NON_b in NON.items():
                    NON_g[repr(b)] = NON_b
                x_rot_g['NR'] = NR
                x_rot_g['K_nn'] = K_nn
        return TpsSolver(bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef)
