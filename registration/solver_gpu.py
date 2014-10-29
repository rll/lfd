from __future__ import division

import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
 
from tpsopt.culinalg_exts import gemm, get_gpu_ptrs, dot_batch_nocheck
import scipy.linalg

class TpsGpuSolver(object):
    """
    class to fit a thin plate spline to data using precomputed
    matrix products
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef, 
                 QN_gpu = None, WQN_gpu = None, NON_gpu = None, NHN_gpu = None):
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
        ## set up GPU memory
        if QN_gpu is None:
            self.QN_gpu = gpuarray.to_gpu(self.QN)
        else:
            self.QN_gpu = QN_gpu
        if WQN_gpu is None:            
            self.WQN_gpu = gpuarray.zeros_like(self.QN_gpu)
        else:
            self.WQN_gpu = WQN_gpu
        if NON_gpu is None:            
            self.NON_gpu = {}
            for b in bend_coefs:
                self.NON_gpu[b] = gpuarray.to_gpu(self.NON[b])
        else:
            self.NON_gpu = NON_gpu
        if NHN_gpu is None:            
            self.NHN_gpu = gpuarray.zeros_like(self.NON_gpu[bend_coefs[0]])
        else:
            self.NHN_gpu = NHN_gpu
        self.valid = True
    
    def _initialize_solver(self, b, wt_n):
        drv.memcpy_dtod_async(self.NHN_gpu.gpudata, self.NON_gpu[b].gpudata,
                              self.NHN_gpu.nbytes)
        self.WQN_gpu.set_async(wt_n[:, None] * self.QN)
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        if bend_coef not in self.bend_coefs:
            raise RuntimeError("No precomputed NON for bending coefficient {}".format(bend_coef))
        assert self.valid
        self._initialize_solver(bend_coef, wt_n)
        gemm(self.QN_gpu, self.WQN_gpu, self.NHN_gpu, 
             transa='T', alpha=1, beta=1)
        lhs = self.NHN_gpu.get()
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = scipy.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class TpsGpuSolverFactory(object):
    """
    pre-allocates the GPU space needed to get a new solver
    efficiently computes solution params and returns a TPSSolver
    """
    def __init__(self, max_N, max_n_iter):
        d = 3
        self.max_N = max_N
        self.max_n_iter = max_n_iter
        self.cur_solver = None
        self.NON_gpu = gpuarray.empty(max_N * max_N * max_n_iter, np.float64)
        self.NHN_gpu = gpuarray.empty(max_N * max_N , np.float64)
        self.QN_gpu = gpuarray.empty(max_N * max_N, np.float64)
        self.WQN_gpu = gpuarray.empty(max_N * max_N, np.float64)
        # temporary space to compute NON
        self.ON_gpu = gpuarray.empty(max_N * (max_N + d + 1)* max_n_iter, np.float64)
        self.O_gpu = gpuarray.empty((max_N +d+1)*(max_N+d+1)* max_n_iter, np.float64)
        self.N_gpu = gpuarray.empty((max_N +d+1)*(max_N) *max_n_iter, np.float64)
    
    def get_solver(self, x_na, K_nn, bend_coefs, rot_coef):
        n,d = x_na.shape
        assert len(bend_coefs) <= self.max_n_iter
        assert n <= self.max_N

        if not self.cur_solver is None:
            self.cur_solver.valid = False

        Q = np.c_[np.ones((n, 1)), x_na, K_nn]
        A = np.r_[np.zeros((d+1, d+1)), np.c_[np.ones((n, 1)), x_na]].T
        
        R = np.zeros((n+d+1, d))
        R[1:d+1, :d] = np.diag(rot_coef)
    
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:].copy()
        N_gpu = self.N_gpu[:(n+d+1)*n].reshape(n+d+1, n)
        N_gpu.set_async(N)
        QN = Q.dot(N)
        QN_gpu = self.QN_gpu[:n*n].reshape(n, n)
        QN_gpu.set_async(QN)
        WQN_gpu = self.WQN_gpu[:n*n].reshape(n, n)
        NHN_gpu = self.NHN_gpu[:n*n].reshape(n, n)
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
            O_gpu.append(self.O_gpu[offset:offset + (n+d+1)*(n+d+1)].reshape(n+d+1, n+d+1))
            O_gpu[-1].set(O_b)

            offset = i * (n)*(n+d+1)
            ON_gpu.append(self.ON_gpu[offset:offset + n*(n+d+1)].reshape(n+d+1, n))
            offset = i * n * n
            NON_gpu.append(self.NON_gpu[offset:offset + n*n].reshape(n, n))
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
        NON_gpu = dict(zip(bend_coefs, NON_gpu))
        NON = dict([(b, non.get_async()) for b, non in NON_gpu.iteritems()])
        self.cur_solver = TpsGpuSolver(bend_coefs, N, QN, NON, NR, x_na, K_nn, rot_coef,
                                    QN_gpu, WQN_gpu, NON_gpu, NHN_gpu)
        return self.cur_solver
