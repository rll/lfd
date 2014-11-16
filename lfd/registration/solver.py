from __future__ import division

import numpy as np
import tps
import os.path
from joblib import Memory

class TpsSolver(object):
    """
    Fits thin plate spline to data using precomputed matrix products
    """
    def __init__(self, N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef):
        self.rot_coef = rot_coef
        self.n, self.d  = x_nd.shape
        self.N          = N
        self.QN         = QN        
        self.NKN        = NKN
        self.NRN        = NRN
        self.NR         = NR
        self.x_nd       = x_nd
        self.K_nn       = K_nn
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        raise NotImplementedError

class TpsSolverFactory(object):
    def __init__(self, use_cache=True, cachedir=None):
        """Inits TpsSolverFactory
        
        Args:
            use_cache: whether to cache solver matrices in file
            cache_dir: cached directory. if not specified, the .cache directory in parent directory of top-level package is used.
        """
        if use_cache:
            if cachedir is None:
                # .cache directory in parent directory of top-level package
                cachedir = os.path.join(__import__(__name__.split('.')[0]).__path__[0], os.path.pardir, ".cache")
            memory = Memory(cachedir=cachedir, verbose=0)
            self.get_solver_mats = memory.cache(self.get_solver_mats)
    
    def get_solver_mats(self, x_nd, rot_coef):
        """Precomputes several of the matrix products needed to fit a TPS exactly.
        A TPS is fit by solving the system:
        N'(Q'WQ + b K + R)N z = -N'(Q'W'y + N'R)
        x = Nz
        where K and R are padded with zeros appropriately.
        
        Returns:
            N, QN, N'KN, N'RN N'R
        """
        raise NotImplementedError
    
    def get_solver(self, x_nd, rot_coef):
        raise NotImplementedError


class CpuTpsSolver(TpsSolver):
    def __init__(self, N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef):
        super(CpuTpsSolver, self).__init__(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        WQN = wt_n[:, None] * self.QN
        lhs = self.QN.T.dot(WQN) + bend_coef * self.NKN + self.NRN
        rhs = self.NR + WQN.T.dot(y_nd)
        z = np.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class CpuTpsSolverFactory(TpsSolverFactory):
    def __init__(self, use_cache=True, cachedir=None):
        super(CpuTpsSolverFactory, self).__init__(use_cache=use_cache, cachedir=cachedir)
    
    def get_solver_mats(self, x_nd, rot_coef):
        n,d = x_nd.shape
        K_nn = tps.tps_kernel_matrix(x_nd)
        A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
        
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:]
        NR = N[1:1+d,:].T * rot_coef
        
        KN = K_nn.dot(N[1+d:,:])
        QN = np.c_[np.ones((n, 1)), x_nd].dot(N[:1+d,:]) + KN
        
        NKN = (N[1+d:,:].T).dot(KN)
        NRN = NR.dot(N[1:1+d,:])
        return N, QN, NKN, NRN, NR, K_nn
    
    def get_solver(self, x_nd, rot_coef):
        N, QN, NKN, NRN, NR, K_nn = self.get_solver_mats(x_nd, rot_coef)
        return CpuTpsSolver(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)

class AutoTpsSolverFactory(TpsSolverFactory):
    def __new__(cls, *args, **kwargs):
        from lfd.registration import _has_cuda
        if _has_cuda:
            from solver_gpu import GpuTpsSolverFactory
            new_instance = object.__new__(GpuTpsSolverFactory, *args, **kwargs)
        else:
            new_instance = object.__new__(CpuTpsSolverFactory, *args, **kwargs)
        return new_instance
