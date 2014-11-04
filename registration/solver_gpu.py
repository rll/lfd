from __future__ import division

import numpy as np
import tps
from solver import TpsSolver, TpsSolverFactory

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import scikits.cuda.linalg as culinalg
culinalg.init()
from tpsopt.culinalg_exts import gemm, get_gpu_ptrs, dot_batch_nocheck

class TpsGpuSolver(TpsSolver):
    """
    Fits thin plate spline to data using precomputed matrix products and doing some operations on the gp
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef, 
                 sqrtWQN_gpu = None, NON_gpu = None, NHN_gpu = None):
        super(TpsGpuSolver, self).__init__(bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef)
        ## set up GPU memory
        if sqrtWQN_gpu is None:
            self.sqrtWQN_gpu = gpuarray.empty(self.QN.shape, np.float64)
        else:
            self.sqrtWQN_gpu = sqrtWQN_gpu
        if NON_gpu is None:
            self.NON_gpu = {}
            for b in bend_coefs:
                self.NON_gpu[b] = gpuarray.to_gpu(self.NON[b])
        else:
            self.NON_gpu = NON_gpu
        if NHN_gpu is None:            
            self.NHN_gpu = gpuarray.empty_like(self.NON_gpu[bend_coefs[0]])
        else:
            self.NHN_gpu = NHN_gpu
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        if bend_coef not in self.bend_coefs:
            raise RuntimeError("No precomputed NON for bending coefficient {}".format(bend_coef))
        drv.memcpy_dtod_async(self.NHN_gpu.gpudata, self.NON_gpu[bend_coef].gpudata,
                              self.NHN_gpu.nbytes)
        self.sqrtWQN_gpu.set_async(np.sqrt(wt_n)[:, None] * self.QN)
        gemm(self.sqrtWQN_gpu, self.sqrtWQN_gpu, self.NHN_gpu, 
             transa='T', alpha=1, beta=1)
        lhs = self.NHN_gpu.get()
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = np.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class TpsGpuSolverFactory(TpsSolverFactory):
    def __init__(self, max_N=None, max_n_iter=None, use_cache=True, cachedir=None):
        """Inits TpsGpuSolverFactory
        
        Args:
            max_N: maximum cloud size for the clouds given to get_solver
            max_n_iter: maximum number of bending coefficients for the ones given to get_solver
            use_cache: whether to cache solver matrices in file
            cache_dir: cached directory. if not specified, the .cache directory in parent directory of top-level package is used.
        
        Note: if max_N and max_n_iter are specified, some gpu arrays are allocated once at construction time, but this doesn't provide significant speed improvement
        """
        super(TpsGpuSolverFactory, self).__init__(use_cache=use_cache, cachedir=cachedir)
        d = 3
        
        self.NON_gpu = None
        
        self.max_N = max_N
        self.max_n_iter = max_n_iter
        if max_N is not None and max_n_iter is not None:
            # temporary space to compute NON
            self.ON_gpu = gpuarray.empty(max_N * (max_N + d + 1)* max_n_iter, np.float64)
            self.O_gpu = gpuarray.empty((max_N +d+1)*(max_N+d+1)* max_n_iter, np.float64)
            self.N_gpu = gpuarray.empty((max_N +d+1)*(max_N) *max_n_iter, np.float64)
        else:
            self.ON_gpu = None
            self.O_gpu = None
            self.N_gpu = None
    
    def get_solver_mats(self, x_na, bend_coefs, rot_coef):
        n,d = x_na.shape
        if self.max_N is not None and n > self.max_N:
            raise RuntimeError("The cloud size is {} but the maximum is {}".format(n, self.max_N))
        if self.max_n_iter is not None and len(bend_coefs) > self.max_N:
            raise RuntimeError("{} bending coefficients are given but the maximum is {}".format(len(bend_coefs), self.max_n_iter))
        
        K_nn = tps.tps_kernel_matrix(x_na)
        Q = np.c_[np.ones((n, 1)), x_na, K_nn]
        A = np.r_[np.zeros((d+1, d+1)), np.c_[np.ones((n, 1)), x_na]].T
        
        R = np.zeros((n+d+1, d))
        R[1:d+1, :d] = np.diag(rot_coef)
    
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:].copy()
        if self.N_gpu is None:
            N_gpu = gpuarray.to_gpu(N)
        else:
            N_gpu = self.N_gpu[:(n+d+1)*n].reshape(n+d+1, n)
            N_gpu.set_async(N)
        QN = Q.dot(N)
        NR = N.T.dot(R)
        
        N_arr_gpu = []
        O_gpu = []
        ON_gpu = []
        NON_gpu = []
        for i, b in enumerate(bend_coefs):
            O_b = np.zeros((n+d+1, n+d+1), np.float64)
            O_b[d+1:, d+1:] += b * K_nn
            O_b[1:d+1, 1:d+1] += np.diag(rot_coef)
            offset = i * (n+d+1)*(n+d+1)
            if self.O_gpu is None:
                O_b_gpu = gpuarray.to_gpu(O_b)
            else:
                O_b_gpu = self.O_gpu[offset:offset + (n+d+1)*(n+d+1)].reshape(n+d+1, n+d+1)
                O_b_gpu.set(O_b)
            O_gpu.append(O_b_gpu)
            offset = i * (n)*(n+d+1)
            if self.ON_gpu is None:
                ON_b_gpu = gpuarray.empty((n+d+1, n), np.float64)
            else:
                ON_b_gpu = self.ON_gpu[offset:offset + n*(n+d+1)].reshape(n+d+1, n)
            ON_gpu.append(ON_b_gpu)
            NON_b_gpu = gpuarray.empty((n, n), np.float64)
            NON_gpu.append(NON_b_gpu)
            N_arr_gpu.append(N_gpu)
        O_ptrs = get_gpu_ptrs(O_gpu)
        ON_ptrs = get_gpu_ptrs(ON_gpu)
        NON_ptrs = get_gpu_ptrs(NON_gpu)
        N_ptrs = get_gpu_ptrs(N_arr_gpu)
        
        dot_batch_nocheck(O_gpu,  N_arr_gpu, ON_gpu,
                          O_ptrs, N_ptrs,    ON_ptrs,
                          b = 0)
        dot_batch_nocheck(N_arr_gpu, ON_gpu,  NON_gpu,
                          N_ptrs,    ON_ptrs, NON_ptrs,
                          transa='T', b = 0)
        self.NON_gpu = dict(zip(bend_coefs, NON_gpu))
        NON = dict([(b, non.get_async()) for b, non in self.NON_gpu.iteritems()])
        
        return N, QN, NON, NR, K_nn
    
    def get_solver(self, x_nd, bend_coefs, rot_coef):
        self.NON_gpu = None # remains as None if the cache is hit
        N, QN, NON, NR, K_nn = self.get_solver_mats_cached(x_nd, bend_coefs, rot_coef)
        return TpsGpuSolver(bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef, NON_gpu=self.NON_gpu)
