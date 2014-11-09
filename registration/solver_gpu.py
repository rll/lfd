from __future__ import division

import numpy as np
import tps
from solver import TpsSolver, TpsSolverFactory

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import scikits.cuda.linalg as culinalg
culinalg.init()
from tpsopt.culinalg_exts import gemm

class TpsGpuSolver(TpsSolver):
    """
    Fits thin plate spline to data using precomputed matrix products and doing some operations on the gp
    """
    def __init__(self, N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef, 
                 sqrtWQN_gpu = None, NKN_gpu = None, NHN_gpu = None):
        super(TpsGpuSolver, self).__init__(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)
        ## set up GPU memory
        if sqrtWQN_gpu is None:
            self.sqrtWQN_gpu = gpuarray.empty(self.QN.shape, np.float64)
        else:
            self.sqrtWQN_gpu = sqrtWQN_gpu
        if NKN_gpu is None:
            self.NKN_gpu = gpuarray.to_gpu(NKN)
        else:
            self.NKN_gpu = NKN_gpu
        if NHN_gpu is None:
            self.NHN_gpu = gpuarray.empty_like(self.NKN_gpu)
        else:
            self.NHN_gpu = NHN_gpu
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        drv.memcpy_dtod_async(self.NHN_gpu.gpudata, self.NKN_gpu.gpudata, self.NHN_gpu.nbytes)
        self.sqrtWQN_gpu.set_async(np.sqrt(wt_n)[:, None] * self.QN)
        gemm(self.sqrtWQN_gpu, self.sqrtWQN_gpu, self.NHN_gpu, 
             transa='T', alpha=1, beta=bend_coef)
        lhs = self.NHN_gpu.get() + self.NRN
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = np.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class TpsGpuSolverFactory(TpsSolverFactory):
    def __init__(self, use_cache=True, cachedir=None):
        """Inits TpsGpuSolverFactory
        
        Args:
            use_cache: whether to cache solver matrices in file
            cache_dir: cached directory. if not specified, the .cache directory in parent directory of top-level package is used.
        """
        super(TpsGpuSolverFactory, self).__init__(use_cache=use_cache, cachedir=cachedir)
    
    def get_solver_mats(self, x_nd, rot_coef):
        n,d = x_nd.shape
        K_nn = tps.tps_kernel_matrix(x_nd)
        A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
        
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:].copy()
        NR = N[1:1+d,:].T * rot_coef
        
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
        return TpsGpuSolver(N, QN, NKN, NRN, NR, x_nd, K_nn, rot_coef)
