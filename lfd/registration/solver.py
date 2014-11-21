from __future__ import division

import numpy as np
import tps
import os.path
from joblib import Memory

import lfd.registration
if lfd.registration._has_cuda:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import scikits.cuda.linalg as culinalg
    from lfd.tpsopt.culinalg_exts import gemm, geam

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
        f_res.update(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta, N=self.N, z=z)


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


class GpuTpsSolver(TpsSolver):
    def __init__(self, N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef):
        if not lfd.registration._has_cuda:
            raise NotImplementedError("CUDA not installed")
        super(GpuTpsSolver, self).__init__(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)
        # the GPU cho_solve requires matrices to be f-contiguous when rhs is a matrix
        self.QN = self.QN.copy(order='F')
        self.sqrtWQN_gpu = gpuarray.empty(self.QN.shape, np.float64, order='F')
        self.NKN_gpu = gpuarray.to_gpu(self.NKN.copy(order='F'))
        self.NRN_gpu = gpuarray.to_gpu(self.NRN.copy(order='F'))
        self.lhs_gpu = gpuarray.empty(self.NKN_gpu.shape, np.float64, order='F')
        self.QN_gpu = gpuarray.to_gpu(self.QN)
        self.NR_gpu = gpuarray.to_gpu(self.NR.copy(order='F'))
        self.y_dnW_gpu = gpuarray.empty(self.x_nd.T.shape, np.float64, order='F')
        self.rhs_gpu = gpuarray.empty(self.NR_gpu.shape, np.float64, order='F')
        if lfd.registration._has_cula:
            self.N_gpu = gpuarray.to_gpu(self.N.copy(order='F'))
            self.theta_gpu = gpuarray.empty((1+self.d+self.n, self.d), np.float64, order='F')
        else:
            self.N_gpu = None
            self.theta_gpu = None
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        if not y_nd.flags.c_contiguous:
            raise RuntimeError("Expected y_nd to be c-contiguous but it isn't")
        self.sqrtWQN_gpu.set_async(np.sqrt(wt_n)[:,None] * self.QN)
        geam(self.NKN_gpu, self.NRN_gpu, self.lhs_gpu, alpha=bend_coef, beta=1)
        gemm(self.sqrtWQN_gpu, self.sqrtWQN_gpu, self.lhs_gpu, transa='T', alpha=1, beta=1)

        drv.memcpy_dtod_async(self.rhs_gpu.gpudata, self.NR_gpu.gpudata, self.rhs_gpu.nbytes)
        self.y_dnW_gpu.set_async(y_nd.T * wt_n) # use transpose so that it is f_contiguous
        gemm(self.QN_gpu, self.y_dnW_gpu, self.rhs_gpu, transa='T', transb='T', alpha=1, beta=1)
        
        if lfd.registration._has_cula:
            culinalg.cho_solve(self.lhs_gpu, self.rhs_gpu)
            z = self.rhs_gpu.get()
            culinalg.dot(self.N_gpu, self.rhs_gpu, out=self.theta_gpu)
            theta = self.theta_gpu.get()
        else: # if cula is not install perform the last two computations in the CPU
            z = np.linalg.solve(self.lhs_gpu.get(), self.rhs_gpu.get())
            theta = self.N.dot(z)
        f_res.update(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta, N=self.N, z=z)


class GpuTpsSolverFactory(TpsSolverFactory):
    def __init__(self, use_cache=True, cachedir=None):
        if not lfd.registration._has_cuda:
            raise NotImplementedError("CUDA not installed")
        super(GpuTpsSolverFactory, self).__init__(use_cache=use_cache, cachedir=cachedir)
    
    def get_solver_mats(self, x_nd, rot_coef):
        n,d = x_nd.shape
        K_nn = tps.tps_kernel_matrix(x_nd)
        A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
        
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:].copy()
        NR = (N[1:1+d,:].T * rot_coef).copy() # so that it is c-contiguous
        
        N_gpu = gpuarray.to_gpu(N[1+d:,:])
        K_gpu = gpuarray.to_gpu(K_nn)
        KN_gpu = culinalg.dot(K_gpu, N_gpu)
        QN = np.c_[np.ones((n, 1)), x_nd].dot(N[:1+d,:]) + KN_gpu.get()
        
        NKN_gpu = culinalg.dot(N_gpu, KN_gpu, transa='T')
        NKN = NKN_gpu.get()
        
        NRN = NR.dot(N[1:1+d,:])
        return N, QN, NKN, NRN, NR, K_nn
    
    def get_solver(self, x_nd, rot_coef):
        N, QN, NKN, NRN, NR, K_nn = self.get_solver_mats(x_nd, rot_coef)
        return GpuTpsSolver(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)


class AutoTpsSolverFactory(TpsSolverFactory):
    def __new__(cls, *args, **kwargs):
        from lfd.registration import _has_cuda
        if _has_cuda:
            new_instance = object.__new__(GpuTpsSolverFactory, *args, **kwargs)
        else:
            new_instance = object.__new__(CpuTpsSolverFactory, *args, **kwargs)
        return new_instance
