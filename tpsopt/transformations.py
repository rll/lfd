import tps
import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
 
from culinalg_exts import gemm, get_gpu_ptrs, dot_batch_nocheck
from cuda_funcs import check_cuda_err
import scipy.linalg

class NoGPUTPSSolver(object):
    """
    class to fit a thin plate spline to data using precomputed
    matrix products
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef = np.r_[1e-4, 1e-4, 1e-1]):
        for b in bend_coefs:
            assert b in NON, 'no solver found for bending coefficient {}'.format(b)
        self.rot_coef = rot_coef
        self.n, self.d  = x_nd.shape
        self.bend_coefs = bend_coefs
        self.N          = N
        self.QN         = QN        
        self.NON        = NON
        self.NR         = NR
        self.x_nd       = x_nd
        self.K_nn       = K_nn
        self.valid = True
    # @profile
    def solve(self, wt_n, y_nd, bend_coef, rot_coef,f_res):
        assert y_nd.shape == (self.n, self.d)
        assert bend_coef in self.bend_coefs
        assert np.allclose(rot_coef, self.rot_coef)
        assert self.valid
        WQN = wt_n[:, None] * self.QN
        lhs = self.NON[b] + self.QN.T.dot(WQN)
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = scipy.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        set_ThinPlateSpline(f_res, self.x_nd, theta)

    @staticmethod
    def get_solvers(h5file):
        solvers = {}
        for seg_name, seg_info in h5file.iteritems():
            solver_info = seg_info['solver']
            N    = solver_info['N'][:]
            QN   = solver_info['QN'][:]
            NR   = solver_info['NR'][:]
            x_nd = solver_info['x_nd'][:]
            K_nn = solver_info['K_nn'][:]
            bend_coefs = [float(x) for x in solver_info['NON'].keys()]
            NON = {}
            for b in bend_coefs:
                NON[b] = solver_info['NON'][str(b)][:]
            solvers[seg_name] = NoGPUTPSSolver(bend_coefs, N, QN, NON, NR, x_nd, K_nn)
        return solvers

class NoGPUEmptySolver(object):
    """
    computes solution params and returns a NoGPUTPSSolver
    """
    def __init__(self, max_N, bend_coefs):
        d = 3
        self.max_N = max_N
        self.bend_coefs = bend_coefs
        self.cur_solver = None
    # @profile
    def get_solver(self, x_na, K_nn, bend_coefs, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
        n,d = x_na.shape
        assert len(bend_coefs) <= len(self.bend_coefs)
        assert n <= self.max_N

        if not self.cur_solver is None:
            self.cur_solver.valid = False

        Q = np.c_[np.ones((n, 1)), x_na, K_nn]
        A = np.r_[np.zeros((d+1, d+1)), np.c_[np.ones((n, 1)), x_na]].T
        
        R = np.zeros((n+d+1, d))
        R[1:d+1, :d] = np.diag(rot_coef)
    
        n_cnts = A.shape[0]    
        _u,_s,_vh = np.linalg.svd(A.T)
        N = _u[:,n_cnts:]
        QN = Q.dot(N)
        NR = N.T.dot(R)
        
        NON = {}
        for i, b in enumerate(bend_coefs):
            O_b = np.zeros((n+d+1, n+d+1), np.float64)
            O_b[d+1:, d+1:] += b * K_nn
            O_b[1:d+1, 1:d+1] += np.diag(rot_coef)
            NON[b] = N.T.dot(O_b.dot(N))
       
        self.cur_solver = NoGPUTPSSolver(bend_coefs, N, QN, NON, NR, x_na, K_nn, rot_coef)
        return self.cur_solver

class TPSSolver(object):
    """
    class to fit a thin plate spline to data using precomputed
    matrix products
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef = np.r_[1e-4, 1e-4, 1e-1], 
                 QN_gpu = None, WQN_gpu = None, NON_gpu = None, NHN_gpu = None):
        for b in bend_coefs:
            assert b in NON, 'no solver found for bending coefficient {}'.format(b)
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
    # @profile
    def initialize_solver(self, b, wt_n):
        drv.memcpy_dtod_async(self.NHN_gpu.gpudata, self.NON_gpu[b].gpudata,
                              self.NHN_gpu.nbytes)
        self.WQN_gpu.set_async(wt_n[:, None] * self.QN)
    # @profile
    def solve(self, wt_n, y_nd, bend_coef, rot_coef,f_res):
        assert y_nd.shape == (self.n, self.d)
        assert bend_coef in self.bend_coefs
        assert np.allclose(rot_coef, self.rot_coef)
        assert self.valid
        self.initialize_solver(bend_coef, wt_n)
        gemm(self.QN_gpu, self.WQN_gpu, self.NHN_gpu, 
             transa='T', alpha=1, beta=1)
        lhs = self.NHN_gpu.get()
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = scipy.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        set_ThinPlateSpline(f_res, self.x_nd, theta)

    @staticmethod
    def get_solvers(h5file):
        solvers = {}
        for seg_name, seg_info in h5file.iteritems():
            solver_info = seg_info['solver']
            N    = solver_info['N'][:]
            QN   = solver_info['QN'][:]
            NR   = solver_info['NR'][:]
            x_nd = solver_info['x_nd'][:]
            K_nn = solver_info['K_nn'][:]
            bend_coefs = [float(x) for x in solver_info['NON'].keys()]
            NON = {}
            for b in bend_coefs:
                NON[b] = solver_info['NON'][str(b)][:]
            solvers[seg_name] = TPSSolver(bend_coefs, N, QN, NON, NR, x_nd, K_nn)
        return solvers        

class EmptySolver(object):
    """
    pre-allocates the GPU space needed to get a new solver
    efficiently computes solution params and returns a TPSSolver
    """
    def __init__(self, max_N, bend_coefs):
        d = 3
        self.max_N = max_N
        self.bend_coefs = bend_coefs
        self.cur_solver = None
        self.NON_gpu = gpuarray.empty(max_N * max_N * len(bend_coefs), np.float64)
        self.NHN_gpu = gpuarray.empty(max_N * max_N , np.float64)
        self.QN_gpu = gpuarray.empty(max_N * max_N, np.float64)
        self.WQN_gpu = gpuarray.empty(max_N * max_N, np.float64)
        # temporary space to compute NON
        self.ON_gpu = gpuarray.empty(max_N * (max_N + d + 1)* len(bend_coefs), np.float64)
        self.O_gpu = gpuarray.empty((max_N +d+1)*(max_N+d+1)* len(bend_coefs), np.float64)
        self.N_gpu = gpuarray.empty((max_N +d+1)*(max_N) *len(bend_coefs), np.float64)
    # @profile
    def get_solver(self, x_na, K_nn, bend_coefs, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
        n,d = x_na.shape
        assert len(bend_coefs) <= len(self.bend_coefs)
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
        self.cur_solver = TPSSolver(bend_coefs, N, QN, NON, NR, x_na, K_nn, rot_coef,
                                    QN_gpu, WQN_gpu, NON_gpu, NHN_gpu)
        return self.cur_solver

class Transformation(object):
    """
    Object oriented interface for transformations R^d -> R^d
    """
    def transform_points(self, x_ma):
        raise NotImplementedError
    def compute_jacobian(self, x_ma):
        raise NotImplementedError        

        
    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])
        

        if orthogonalize:
            if orth_method == "qr": 
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
        
    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        hmat_mGD[:,3,:] = np.array([0,0,0,1])
        return hmat_mGD
        
    def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
        "numerical jacobian"
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros(len(x0), len(f0))
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
            dx[i] = 0.
        return jac.transpose()

class ThinPlateSpline(Transformation):
    """
    members:
        x_na: centers of basis functions
        w_ng: 
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    
    """
    def __init__(self, d=3):
        "initialize as identity"
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))

    def transform_points(self, x_ma):
        y_ng = tps.tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return y_ng
    def compute_jacobian(self, x_ma):
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
        
class Affine(Transformation):
    def __init__(self, lin_ag, trans_g):
        self.lin_ag = lin_ag
        self.trans_g = trans_g
    def transform_points(self, x_ma):
        return x_ma.dot(self.lin_ag) + self.trans_g[None,:]  
    def compute_jacobian(self, x_ma):
        return np.repeat(self.lin_ag.T[None,:,:],len(x_ma), axis=0)
        
class Composition(Transformation):
    def __init__(self, fs):
        "applied from first to last (left to right)"
        self.fs = fs
    def transform_points(self, x_ma):
        for f in self.fs: x_ma = f.transform_points(x_ma)
        return x_ma
    def compute_jacobian(self, x_ma):
        grads = []
        for f in self.fs:
            grad_mga = f.compute_jacobian(x_ma)
            grads.append(grad_mga)
            x_ma = f.transform_points(x_ma)
        totalgrad = grads[0]
        for grad in grads[1:]:
            totalgrad = (grad[:,:,:,None] * totalgrad[:,None,:,:]).sum(axis=-2)
        return totalgrad

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    f = ThinPlateSpline()
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.x_na = x_na
    return f

def set_ThinPlateSpline(f, x_na, theta):
    f.x_na = x_na
    d = x_na.shape[1]
    f.trans_g = theta[0]
    f.lin_ag  = theta[1:d+1]
    f.w_ng    = theta[d+1:]
