#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
import h5py, sys

import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray
from scikits.cuda import linalg
linalg.init()

from tps import tps_kernel_matrix, tps_eval
from transformations import unit_boxify
from culinalg_exts import dot_batch_nocheck, get_gpu_ptrs, m_dot_batch
from precompute import downsample_cloud, batch_get_sol_params
from cuda_funcs import init_prob_nm, norm_prob_nm, get_targ_pts, check_cuda_err, fill_mat, reset_cuda, sq_diffs
from registration import registration_cost as cpu_registration_cost
from constants import N_ITER_CHEAP, EM_ITER_CHEAP, DEFAULT_LAMBDA, MAX_CLD_SIZE, DATA_DIM, DS_SIZE, N_STREAMS, DEFAULT_NORM_ITERS, BEND_COEF_DIGITS, MAX_TRAJ_LEN

import IPython as ipy
from pdb import pm, set_trace
import time


class Globals:
    sync = False
    streams = []
    for i in range(N_STREAMS):
        streams.append(drv.Stream())

def get_stream(i):
    return Globals.streams[i % N_STREAMS]

def sync(override = False):
    if Globals.sync or override:
        check_cuda_err()

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

def gpu_pad(x, shape, dtype=np.float32):
    (m, n) = x.shape
    if m > shape[0] or n > shape[1]:
        raise ValueError("Cannot Pad Beyond Normal Dimension")
    x_new = np.zeros(shape, dtype=dtype)    
    x_new[:m, :n] = x
    return gpuarray.to_gpu(x_new)

class GPUContext(object):
    """
    Class to contain GPU arrays
    """
    def __init__(self, bend_coefs = None):
        if bend_coefs is None:
            lambda_init, lambda_final = DEFAULT_LAMBDA
            bend_coefs = np.around(loglinspace(lambda_init, lambda_final, N_ITER_CHEAP), 
                                    BEND_COEF_DIGITS)
        self.bend_coefs = bend_coefs
        self.ptrs_valid = False
        self.N = 0
        self.ptr_list = []

        self.tps_params     = []
        self.tps_param_ptrs = None
        self.trans_d        = []
        self.trans_d_ptrs   = None
        self.lin_dd         = []
        self.lin_dd_ptrs    = None
        self.w_nd           = []
        self.w_nd_ptrs      = None
        """
        TPS PARAM FORMAT
        [      np.zeros(DATA_DIM)      ]   [  trans_d  ]   [1 x d]
        [       np.eye(DATA_DIM)       ] = [  lin_dd   ] = [d x d]
        [np.zeros((np.zeros, DATA_DIM))]   [   w_nd    ]   [n x d]
        """
        self.default_tps_params = gpuarray.zeros((DATA_DIM + 1 + MAX_CLD_SIZE, DATA_DIM), np.float32)
        self.default_tps_params[1:DATA_DIM+1, :].set(np.eye(DATA_DIM, dtype=np.float32))

        self.proj_mats       = dict([(b, []) for b in bend_coefs])
        self.proj_mat_ptrs   = dict([(b, None) for b in bend_coefs])
        self.offset_mats     = dict([(b, []) for b in bend_coefs])
        self.offset_mat_ptrs = dict([(b, None) for b in bend_coefs])
        self.pts          = []
        self.pt_ptrs      = None
        self.kernels      = []
        self.kernel_ptrs  = None
        self.pts_w        = []
        self.pt_w_ptrs    = None
        self.pts_t        = []
        self.pt_t_ptrs    = None
        self.dims         = []
        self.dims_gpu     = None
        self.scale_params = []


        self.warp_err      = None
        self.bend_res      = []
        self.bend_res_ptrs = None

        self.corr_cm        = []
        self.corr_cm_ptrs   = None
        self.corr_rm        = []
        self.corr_rm_ptrs   = None
        self.r_coefs        = []
        self.r_coef_ptrs    = None
        self.c_coefs_rn     = []
        self.c_coef_rn_ptrs = None
        self.c_coefs_cn     = []
        self.c_coef_cn_ptrs = None

        self.seg_names       = []

    def reset_tps_params(self):
        """
        sets the tps params to be identity
        """
        for p in self.tps_params:
            drv.memcpy_dtod_async(p.gpudata, self.default_tps_params.gpudata, p.nbytes)            
    def set_tps_params(self, vals):
        for d, s in zip(self.tps_params, vals):
            drv.memcpy_dtod_async(d.gpudata, s.gpudata, d.nbytes)            

    def reset_warp_err(self):
        self.warp_err.fill(0)

    def check_cld(self, cloud_xyz):
        if cloud_xyz.dtype != np.float32:
            raise TypeError("only single precision operations supported")
        if cloud_xyz.shape[0] > MAX_CLD_SIZE:
            raise ValueError("cloud size exceeds {}".format(MAX_CLD_SIZE))
        if cloud_xyz.shape[1] != DATA_DIM:
            raise ValueError("point cloud must have cumn dimension {}".format(DATA_DIM))
    # @profile
    def get_sol_params(self, cld):
        self.check_cld(cld)
        K = tps_kernel_matrix(cld)
        proj_mats   = {}
        offset_mats = {}
        (proj_mats_arr, _), (offset_mats_arr, _) = batch_get_sol_params(cld, K, self.bend_coefs)
        for i, b in enumerate(self.bend_coefs):
            proj_mats[b]   = proj_mats_arr[i]
            offset_mats[b] = offset_mats_arr[i]
        return proj_mats, offset_mats, K

    def add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, scale_params, update_ptrs = False):
        """
        adds a new cloud to our context for batch processing
        """
        self.check_cld(cloud_xyz)
        self.ptrs_valid = False
        self.N += 1
        self.seg_names.append(name)
        self.tps_params.append(self.default_tps_params.copy())
        self.trans_d.append(self.tps_params[-1][0, :])
        self.lin_dd.append(self.tps_params[-1][1:DATA_DIM+1, :])
        self.w_nd.append(self.tps_params[-1][DATA_DIM + 1:, :])
        self.scale_params.append(scale_params)
        n = cloud_xyz.shape[0]
        
        for b in self.bend_coefs:
            proj_mat   = proj_mats[b]
            offset_mat = offset_mats[b]
            self.proj_mats[b].append(gpu_pad(proj_mat, (MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE)))

            if offset_mat.shape != (n + DATA_DIM + 1, DATA_DIM):
                raise ValueError("Offset Matrix has incorrect dimension")
            self.offset_mats[b].append(gpu_pad(offset_mat, (MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM)))


        if n > MAX_CLD_SIZE or cloud_xyz.shape[1] != DATA_DIM:
            raise ValueError("cloud_xyz has incorrect dimension")
        self.pts.append(gpu_pad(cloud_xyz, (MAX_CLD_SIZE, DATA_DIM)))
        if kernel.shape != (n, n):
            raise ValueError("dimension mismatch b/t kernel and cloud")
        self.kernels.append(gpu_pad(kernel, (MAX_CLD_SIZE, MAX_CLD_SIZE)))
        self.dims.append(n)

        self.pts_w.append(gpuarray.zeros_like(self.pts[-1]))
        self.pts_t.append(gpuarray.zeros_like(self.pts[-1]))
        self.corr_cm.append(gpuarray.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32))
        self.corr_rm.append(gpuarray.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32))
        self.r_coefs.append(gpuarray.zeros((MAX_CLD_SIZE, 1), np.float32))
        self.c_coefs_rn.append(gpuarray.zeros((MAX_CLD_SIZE, 1), np.float32))
        self.c_coefs_cn.append(gpuarray.zeros((MAX_CLD_SIZE, 1), np.float32))

        if update_ptrs:
            self.update_ptrs()

    def update_ptrs(self):
        self.tps_param_ptrs = get_gpu_ptrs(self.tps_params)
        self.trans_d_ptrs   = get_gpu_ptrs(self.trans_d)
        self.lin_dd_ptrs    = get_gpu_ptrs(self.lin_dd)
        self.w_nd_ptrs      = get_gpu_ptrs(self.w_nd)
        
        for b in self.bend_coefs:
            self.proj_mat_ptrs[b]   = get_gpu_ptrs(self.proj_mats[b])
            self.offset_mat_ptrs[b] = get_gpu_ptrs(self.offset_mats[b])

        self.pt_ptrs        = get_gpu_ptrs(self.pts)
        self.kernel_ptrs    = get_gpu_ptrs(self.kernels)
        self.pt_w_ptrs      = get_gpu_ptrs(self.pts_w)
        self.pt_t_ptrs      = get_gpu_ptrs(self.pts_t)
        self.corr_cm_ptrs   = get_gpu_ptrs(self.corr_cm)
        self.corr_rm_ptrs   = get_gpu_ptrs(self.corr_rm)
        self.r_coef_ptrs    = get_gpu_ptrs(self.r_coefs)
        self.c_coef_rn_ptrs = get_gpu_ptrs(self.c_coefs_rn)
        self.c_coef_cn_ptrs = get_gpu_ptrs(self.c_coefs_cn)
        # temporary space for warping cost computations
        self.warp_err        = gpuarray.zeros((self.N, MAX_CLD_SIZE), np.float32)
        self.bend_res_mat    = gpuarray.zeros((DATA_DIM * self.N, DATA_DIM), np.float32)
        self.bend_res        =[self.bend_res_mat[i*DATA_DIM:(i+1)*DATA_DIM] for i in range(self.N)]
        self.bend_res_ptrs   = get_gpu_ptrs(self.bend_res)

        self.dims_gpu = gpuarray.to_gpu(np.array(self.dims, dtype=np.int32))
        self.ptrs_valid = True


    # @profile
    def setup_tgt_ctx(self, cloud_xyz):
        """
        returns a GPUContext where all the clouds are cloud_xyz
        and matched in length with this contex

        assumes cloud_xyz is already downsampled and scaled
        """        
        tgt_ctx = TgtContext(self)
        tgt_ctx.set_cld(cloud_xyz)
        return tgt_ctx

    # @profile
    def transform_points(self):
        """
        computes the warp of self.pts under the current tps params
        """
        fill_mat(self.pt_w_ptrs, self.trans_d_ptrs, self.dims_gpu, self.N)
        dot_batch_nocheck(self.pts,         self.lin_dd,      self.pts_w,
                          self.pt_ptrs,     self.lin_dd_ptrs, self.pt_w_ptrs) 
        dot_batch_nocheck(self.kernels,     self.w_nd,        self.pts_w,
                          self.kernel_ptrs, self.w_nd_ptrs,   self.pt_w_ptrs) 
        sync()
    
    # @profile
    def get_target_points(self, other, outlierprior=1e-1, outlierfrac=1e-2, outliercutoff=1e-2, 
                          T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
        """
        computes the target points for self and other
        using the current warped points for both                
        """
        init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                     self.pt_w_ptrs, other.pt_w_ptrs, 
                     self.dims_gpu, other.dims_gpu,
                     self.N, outlierprior, outlierfrac, T, 
                     self.corr_cm_ptrs, self.corr_rm_ptrs)
        sync()
        norm_prob_nm(self.corr_cm_ptrs, self.corr_rm_ptrs, 
                     self.dims_gpu, other.dims_gpu, self.N, outlierfrac, norm_iters,
                     self.r_coef_ptrs, self.c_coef_rn_ptrs, self.c_coef_cn_ptrs)        
        sync()
        get_targ_pts(self.pt_ptrs, other.pt_ptrs,
                     self.pt_w_ptrs, other.pt_w_ptrs,
                     self.corr_cm_ptrs, self.corr_rm_ptrs,
                     self.r_coef_ptrs, self.c_coef_rn_ptrs, self.c_coef_cn_ptrs,
                     self.dims_gpu, other.dims_gpu, 
                     outliercutoff, self.N,
                     self.pt_t_ptrs, other.pt_t_ptrs)
        sync()

    # @profile
    def update_transform(self, b):
        """
        computes the TPS associated with the current target pts
        """
        self.set_tps_params(self.offset_mats[b])
        dot_batch_nocheck(self.proj_mats[b],     self.pts_t,     self.tps_params,
                          self.proj_mat_ptrs[b], self.pt_t_ptrs, self.tps_param_ptrs)
        sync()
    # @profile
    def mapping_cost(self, other, bend_coef=DEFAULT_LAMBDA[1], outlierprior=1e-1, outlierfrac=1e-2, 
                       outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
        """
        computes the error in the current mapping
        assumes that the target points have already been filled
        """
        self.transform_points()
        other.transform_points()
        sums = []
        sq_diffs(self.pt_w_ptrs, self.pt_t_ptrs, self.warp_err, self.N, True)
        sq_diffs(other.pt_w_ptrs, other.pt_t_ptrs, self.warp_err, self.N, False)
        warp_err = self.warp_err.get()
        return np.sum(warp_err, axis=1)
    # @profile
    def bending_cost(self, b=DEFAULT_LAMBDA[1]):
        ## b * w_nd' * K * w_nd
        ## use pts_w as temporary storage
        dot_batch_nocheck(self.kernels,     self.w_nd,      self.pts_w,
                          self.kernel_ptrs, self.w_nd_ptrs, self.pt_w_ptrs,
                          b = 0)

        dot_batch_nocheck(self.pts_w,     self.w_nd,      self.bend_res,
                          self.pt_w_ptrs, self.w_nd_ptrs, self.bend_res_ptrs,
                          transa='T', b = 0)
        bend_res = self.bend_res_mat.get()        
        return b * np.array([np.trace(bend_res[i*DATA_DIM:(i+1)*DATA_DIM]) for i in range(self.N)])
    # @profile
    def bidir_tps_cost(self, other, bend_coef=1, outlierprior=1e-1, outlierfrac=1e-2, 
                       outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
        self.reset_warp_err()
        mapping_err  = self.mapping_cost(other, outlierprior, outlierfrac, outliercutoff, T, norm_iters)
        bending_cost = self.bending_cost(bend_coef)
        bending_cost += other.bending_cost(bend_coef)
        return mapping_err + bending_cost

    """
    testing for custom kernels
    """

    def test_mapping_cost(self, other, bend_coef=DEFAULT_LAMBDA[1], outlierprior=1e-1, outlierfrac=1e-2, 
                       outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
        mapping_err = self.mapping_cost(other, outlierprior, outlierfrac, outliercutoff, T, norm_iters)
        for i in range(self.N):
            ## compute error for 0 on cpu
            s_gpu = mapping_err[i]
            s_cpu = np.float32(0)
            xt = self.pts_t[i].get()
            xw = self.pts_w[i].get()
            
            yt = other.pts_t[i].get()
            yw = other.pts_w[i].get()
            
            ##use the trace b/c then numpy will use float32s all the way
            s_cpu += np.trace(xt.T.dot(xt) + xw.T.dot(xw) - 2 * xw.T.dot(xt))
            s_cpu += np.trace(yt.T.dot(yt) + yw.T.dot(yw) - 2 * yw.T.dot(yt))
            
            if not np.isclose(s_cpu, s_gpu, atol=1e-4):
                ## high err tolerance is b/c of difference in cpu and gpu precision?
                print "cpu and gpu sum sq differences differ!!!"
                ipy.embed()
                sys.exit(1)

    def test_bending_cost(self, other, bend_coef=DEFAULT_LAMBDA[1], outlierprior=1e-1, outlierfrac=1e-2, 
                       outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
        self.get_target_points(other, outlierprior, outlierfrac, outliercutoff,  T, norm_iters)
        self.update_transform(bend_coef)
        bending_costs = self.bending_cost(bend_coef)
        for i in range(self.N):
            c_gpu = bending_costs[i]
            k_nn = self.kernels[i].get()
            w_nd = self.w_nd[i].get()
            c_cpu = np.float32(0)
            for d in range(DATA_DIM):
                r = np.dot(k_nn, w_nd[:, d]).astype(np.float32)
                r = np.float32(np.dot(w_nd[:, d], r))
                c_cpu += r
            c_cpu *= np.float32(bend_coef)
            if np.abs(c_cpu - c_gpu) > 1e-4:
                ## high err tolerance is b/c of difference in cpu and gpu precision?
                print "cpu and gpu bend costs differ!!!"
                ipy.embed()
                sys.exit(1)    

    def test_init_corr(self, other, T = 5e-3, outlierprior=1e-1, outlierfrac=1e-2, outliercutoff=1e-2, ):
        import scipy.spatial.distance as ssd
        import sys
        self.transform_points()
        other.transform_points()
        init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                     self.pt_w_ptrs, other.pt_w_ptrs, 
                     self.dims_gpu, other.dims_gpu,
                     self.N, outlierprior, outlierfrac, T, 
                     self.corr_cm_ptrs, self.corr_rm_ptrs)
        gpu_corr_rm = self.corr_rm[0].get()
        gpu_corr_rm = gpu_corr_rm.flatten()[:(self.dims[0] + 1) * (other.dims[0] + 1)].reshape(self.dims[0]+1, other.dims[0]+1)
        s_pt_w = self.pts_w[0].get()
        s_pt   = self.pts[0].get()
        o_pt_w = other.pts_w[0].get()
        o_pt   = other.pts[0].get()

        d1 = ssd.cdist(s_pt_w, o_pt, 'euclidean')
        d2 = ssd.cdist(s_pt, o_pt_w, 'euclidean')

        p_nm = np.exp( -(d1 + d2) / (2 * T))

        for i in range(self.dims[0]):
            for j in range(other.dims[0]):
                if abs(p_nm[i, j] - gpu_corr_rm[i, j]) > 1e-7:
                    print "INIT CORR MATRICES DIFFERENT"
                    print i, j, p_nm[i, j], gpu_corr_rm[i, j]
                    ipy.embed()
                    sys.exit(1)

    def test_norm_corr(self, other, T = 5e-3, outlierprior=1e-1, outlierfrac=1e-2, outliercutoff=1e-2, norm_iters = DEFAULT_NORM_ITERS):
        import sys
        self.transform_points()
        other.transform_points()
        init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                     self.pt_w_ptrs, other.pt_w_ptrs, 
                     self.dims_gpu, other.dims_gpu,
                     self.N, outlierprior, outlierfrac, T, 
                     self.corr_cm_ptrs, self.corr_rm_ptrs)
        n, m  = self.dims[0], other.dims[0]
        init_corr = self.corr_rm[0].get()        
        init_corr = init_corr.flatten()[:(n + 1) * (m + 1)].reshape(n+1, m+1).astype(np.float32)
        
        a_N = np.ones((n+1),dtype = np.float32)
        a_N[n] = m*outlierfrac
        b_M = np.ones((m+1), dtype = np.float32)
        b_M[m] = n*outlierfrac

        old_r_coefs = np.ones(n+1, dtype=np.float32)
        old_c_coefs = np.ones(m+1, dtype=np.float32)
        for n_iter in range(1, norm_iters):
            init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                         self.pt_w_ptrs, other.pt_w_ptrs, 
                         self.dims_gpu, other.dims_gpu,
                         self.N, outlierprior, outlierfrac, T, 
                         self.corr_cm_ptrs, self.corr_rm_ptrs)

            norm_prob_nm(self.corr_cm_ptrs, self.corr_rm_ptrs, 
                         self.dims_gpu, other.dims_gpu, self.N, outlierfrac, n_iter,
                         self.r_coef_ptrs, self.c_coef_rn_ptrs, self.c_coef_cn_ptrs)        
            new_r_coefs = a_N/init_corr.dot(old_c_coefs[:m+1])
            new_c_coefs = b_M/new_r_coefs[:n+1].dot(init_corr)
            gpu_c_coefs = self.c_coefs_cn[0].get().flatten()[:m + 1].reshape(m+1)
            gpu_r_coefs = self.r_coefs[0].get().flatten()[:n + 1].reshape(n+1)
            if not np.allclose(new_r_coefs, gpu_r_coefs):
                print "row coefficients don't match", n_iter
                ipy.embed()
                sys.exit(1)
            if not np.allclose(new_c_coefs, gpu_c_coefs):
                print "column coefficients don't match", n_iter
                ipy.embed()
                sys.exit(1)
            # old_r_coefs = gpu_r_coefs
            # old_c_coefs = gpu_c_coefs
            old_r_coefs = new_r_coefs
            old_c_coefs = new_c_coefs
            
    def test_get_targ(self, other, T = 5e-3, outlierprior=1e-1, outlierfrac=1e-2, outliercutoff=1e-2, norm_iters = DEFAULT_NORM_ITERS):
        self.transform_points()
        other.transform_points()
        init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                     self.pt_w_ptrs, other.pt_w_ptrs, 
                     self.dims_gpu, other.dims_gpu,
                     self.N, outlierprior, outlierfrac, T, 
                     self.corr_cm_ptrs, self.corr_rm_ptrs)
        norm_prob_nm(self.corr_cm_ptrs, self.corr_rm_ptrs, 
                     self.dims_gpu, other.dims_gpu, self.N, outlierfrac, norm_iters,
                     self.r_coef_ptrs, self.c_coef_rn_ptrs, self.c_coef_cn_ptrs)
        get_targ_pts(self.pt_ptrs, other.pt_ptrs,
                     self.pt_w_ptrs, other.pt_w_ptrs,
                     self.corr_cm_ptrs, self.corr_rm_ptrs,
                     self.r_coef_ptrs, self.c_coef_rn_ptrs, self.c_coef_cn_ptrs,
                     self.dims_gpu, other.dims_gpu, 
                     outliercutoff, self.N,
                     self.pt_t_ptrs, other.pt_t_ptrs)
        n, m = self.dims[0], other.dims[0]
        x = self.pts[0].get()[:n]
        xw = self.pts_w[0].get()[:n]
        xt = self.pts_t[0].get()[:n]
        y = other.pts[0].get()[:m]
        yw = other.pts_w[0].get()[:m]
        yt = other.pts_t[0].get()[:m]

        init_corr = self.corr_rm[0].get()        
        init_corr = init_corr.flatten()[:(n + 1) * (m + 1)].reshape(n+1, m+1).astype(np.float32)
        gpu_c_cn_coefs = self.c_coefs_cn[0].get().flatten()[:m + 1].reshape(m+1)
        gpu_c_rn_coefs = self.c_coefs_rn[0].get().flatten()[:m + 1].reshape(m+1)
        gpu_r_coefs = self.r_coefs[0].get().flatten()[:n + 1].reshape(n+1)

        rn_corr = (init_corr * gpu_c_rn_coefs[None, :]) * gpu_r_coefs[:, None]
        cn_corr = (init_corr * gpu_c_cn_coefs[None, :]) * gpu_r_coefs[:, None]        
        rn_corr = rn_corr[:n, :m]
        cn_corr = cn_corr[:n, :m]
        
        wt_n = rn_corr.sum(axis=1)
        inlier = wt_n > outliercutoff
        xtarg = np.empty((n, DATA_DIM))
        xtarg[inlier, :] = rn_corr.dot(y)[inlier, :]
        xtarg[~inlier, :] = xw[~inlier, :]

        if not np.allclose(xtarg, xt):
            print "xt values differ"
            ipy.embed()
            sys.exit(1)

        wt_m = cn_corr.sum(axis=0)
        inliner = wt_m > outliercutoff
        ytarg = np.empty((m, DATA_DIM))
        ytarg[inlier, :] = cn_corr.T.dot(x)[inliner, :]
        ytarg[~inlier, :] = yw[~inlier, :]
        if not np.allclose(ytarg, yt):
            print "yt values differ"
            ipy.embed()
            sys.exit(1)                            

    def unit_test(self, other):
        print "running basic unit tests"
        self.test_init_corr(other)
        self.test_norm_corr(other)
        self.test_get_targ(other)
        self.test_mapping_cost(other)
        self.test_bending_cost(other)
        print "UNIT TESTS PASSED"

class SrcContext(GPUContext):
    """
    specialized class to handle source clouds
    includes support for warped trajectories as well
    """
    def __init__(self, bend_coefs=None):
        GPUContext.__init__(self, bend_coefs)
        """
        items for the trajectory and warping thereof
        """
        self.l_traj          = []
        self.l_traj_ptrs     = None
        self.l_traj_K        = []
        self.l_traj_K_ptrs   = None
        self.l_traj_w        = []
        self.l_traj_w_ptrs   = None
        self.r_traj          = []
        self.r_traj_ptrs     = None
        self.r_traj_K        = []
        self.r_traj_K_ptrs   = None
        self.r_traj_w        = []
        self.r_traj_w_ptrs   = None
        self.ptr_list.extend([(self.l_traj,   self.l_traj_ptrs),
                              (self.l_traj_K, self.l_traj_K_ptrs),
                              (self.l_traj_w, self.l_traj_w_ptrs),
                              (self.r_traj,   self.r_traj_ptrs),
                              (self.r_traj_K, self.r_traj_K_ptrs),
                              (self.r_traj_w, self.r_traj_w_ptrs)])

    def update_ptrs(self):
        self.l_traj_ptrs = get_gpu_ptrs(self.l_traj)
        self.l_traj_K_ptrs = get_gpu_ptrs(self.l_traj_K)
        self.l_traj_w_Ptrs = get_gpu_ptrs(self.l_traj_w)
        self.r_traj_ptrs = get_gpu_ptrs(self.r_traj)
        self.r_traj_K_ptrs = get_gpu_ptrs(self.r_traj_K)
        self.r_traj_w_Ptrs = get_gpu_ptrs(self.r_traj_w)
        GPUContext.update_ptrs(self)

    def add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, scale_params,
                r_traj, r_traj_K, l_traj, l_traj_K, update_ptrs = False):
        """
        does the normal add, but also adds the trajectories too
        """
        # don't update ptrs there, do it after this
        GPUContext.add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, scale_params,
                           update_ptrs=False)
        self.r_traj.append(gpu_pad(r_traj, (MAX_TRAJ_LEN, DATA_DIM)))
        self.r_traj_K.append(gpu_pad(r_traj_K, (MAX_TRAJ_LEN, MAX_CLD_SIZE)))
        self.l_traj.append(gpu_pad(l_traj, (MAX_TRAJ_LEN, DATA_DIM)))
        self.l_traj_K.append(gpu_pad(r_traj_K, (MAX_TRAJ_LEN, MAX_CLD_SIZE)))

        self.r_traj_w.append(gpuarray.zeros_like(self.r_traj[-1]))
        self.l_traj_w.append(gpuarray.zeros_like(self.l_traj[-1]))

        if update_ptrs:
            self.update_ptrs()

    def read_h5(self, fname):
        f = h5py.File(fname, 'r')
        for seg_name, seg_info in f.iteritems():
            if 'inv' not in seg_info:
                raise KeyError("H5 File does not have precomputed values")
            seg_info = seg_info['inv']

            proj_mats   = {}
            offset_mats = {}
            for b in self.bend_coefs:
                k = str(b)
                if k not in seg_info:
                    raise KeyError("H5 File {} bend coefficient {}".format(seg_name, k))
                proj_mats[b] = seg_info[k]['proj_mat'][:]
                offset_mats[b] = seg_info[k]['offset_mat'][:]

            ds_g         = seg_info['DS_SIZE_{}'.format(DS_SIZE)]
            cloud_xyz    = ds_g['scaled_cloud_xyz'][:]
            kernel       = ds_g['scaled_K_nn'][:]
            r_traj       = ds_g['scaled_r_traj'][:]
            r_traj_K     = ds_g['scaled_r_traj_K'][:]
            l_traj       = ds_g['scaled_l_traj'][:]  
            l_traj_K     = ds_g['scaled_l_traj_K'][:]          
            scale_params = (ds_g['scaling'][()], ds_g['scaled_translation'][:])
            self.add_cld(seg_name, proj_mats, offset_mats, cloud_xyz, kernel, scale_params,
                         r_traj, r_traj_K, l_traj, l_traj_K)
        f.close()
        self.update_ptrs()        

class TgtContext(GPUContext):
    """
    specialized class to handle the case where we are
    mapping to a single target cloud --> only allocate GPU Memory once
    """
    def __init__(self, src_ctx):
        GPUContext.__init__(self, src_ctx.bend_coefs)
        self.src_ctx = src_ctx
        ## just setup with 0's
        tgt_cld = np.zeros((MAX_CLD_SIZE, DATA_DIM), np.float32)
        proj_mats = dict([(b, np.zeros((MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE), np.float32)) 
                          for b in self.bend_coefs])
        offset_mats = dict([(b, np.zeros((MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM), np.float32)) 
                            for b in self.bend_coefs])
        tgt_K = np.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32)
        for n in src_ctx.seg_names:
            name = "{}_tgt".format(n)
            GPUContext.add_cld(self, name, proj_mats, offset_mats, tgt_cld, tgt_K, None)
        GPUContext.update_ptrs(self)
    def add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, update_ptrs = False):
        raise NotImplementedError("not implemented for TgtConext")
    def update_ptrs(self):
        raise NotImplementedError("not implemented for TgtConext")
    # @profile
    def set_cld(self, cld):
        """
        sets the cloud for this appropriately
        won't allocate any new memory
        """                          
        scaled_cld, scale_params = unit_boxify(cld)
        proj_mats, offset_mats, K = self.get_sol_params(scaled_cld)
        K_gpu = gpu_pad(K, (MAX_CLD_SIZE, MAX_CLD_SIZE))
        cld_gpu = gpu_pad(scaled_cld, (MAX_CLD_SIZE, DATA_DIM))
        self.pts          = [cld_gpu for _ in range(self.N)]
        self.scale_params = [scale_params for _ in range(self.N)]
        self.kernels      = [K_gpu for _ in range(self.N)]
        proj_mats_gpu     = dict([(b, gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE)))
                                  for b, p in proj_mats.iteritems()])
        self.proj_mats    = dict([(b, [p for _ in range(self.N)])
                                  for b, p in proj_mats_gpu.iteritems()])
        offset_mats_gpu   = dict([(b, gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM))) 
                                  for b, p in offset_mats.iteritems()])
        self.offset_mats  = dict([(b, [p for _ in range(self.N)])
                                  for b, p in offset_mats_gpu.iteritems()])
        self.dims         = [scaled_cld.shape[0]]

        self.pt_ptrs.fill(int(self.pts[0].gpudata))
        self.kernel_ptrs.fill(int(self.kernels[0].gpudata))
        self.dims_gpu.fill(self.dims[0])
        for b in self.bend_coefs:
            self.proj_mat_ptrs[b].fill(int(self.proj_mats[b][0].gpudata))
            self.offset_mat_ptrs[b].fill(int(self.offset_mats[b][0].gpudata))

def check_transform_pts(ctx, i = 0):
    import scikits.cuda.linalg as la
    n = ctx.dims[i]
    w_nd = ctx.w_nd[i].get()[:n]
    lin_dd = ctx.lin_dd[i].get()
    trans_d = ctx.trans_d[i].get()
    k_nn = ctx.kernels[i].get()[:n, :n].reshape(n, n).copy()
    x_nd = ctx.pts[i].get()[:n]
    xw_nd = ctx.pts_w[i].get()[:n]

    _k_gpu = gpuarray.to_gpu(k_nn)
    _x_gpu = gpuarray.to_gpu(x_nd)
    _lin_gpu = gpuarray.to_gpu(lin_dd)
    _trans_gpu = gpuarray.to_gpu(trans_d)
    _w_gpu = gpuarray.to_gpu(w_nd)
    
    fill_mat(ctx.pt_w_ptrs, ctx.trans_d_ptrs, ctx.dims_gpu, ctx.N)
    dot_batch_nocheck(ctx.pts,         ctx.lin_dd,      ctx.pts_w,
                      ctx.pt_ptrs,     ctx.lin_dd_ptrs, ctx.pt_w_ptrs) 

    xw_nd = ctx.pts_w[i].get()[:n]
    cpu_xw_nd = np.dot(x_nd, lin_dd) + trans_d[None, :]
    # assert np.allclose(xw_nd, cpu_xw_nd)

    dot_batch_nocheck(ctx.kernels,     ctx.w_nd,        ctx.pts_w,
                      ctx.kernel_ptrs, ctx.w_nd_ptrs,   ctx.pt_w_ptrs) 
    xw_nd = ctx.pts_w[i].get()[:n]
    cpu_xw_nd = cpu_xw_nd + np.dot(k_nn, w_nd)
    # print "w_nd\n", w_nd[:3], np.max(w_nd)
    # print "lin_dd\n", lin_dd[:3]
    # print "trans_d\n", trans_d
    # print "k_nn\n", k_nn[:3, :3]
    # print "x_nd\n", x_nd[:3, :3]
    # print cpu_xw_nd[:3]
    if not(np.allclose(xw_nd, cpu_xw_nd) ):
        print "k dot w_nd is difference on cpu and gpu"
        k_dot_w = np.dot(k_nn, w_nd)
        k_gpu = [gpuarray.to_gpu(k_nn)]
        w_gpu = [gpuarray.to_gpu(w_nd)]
        res_gpu = [gpuarray.zeros((n, DATA_DIM), np.float32)]        
        k_ptrs = get_gpu_ptrs(k_gpu)
        w_ptrs = get_gpu_ptrs(w_gpu)
        res_ptrs = get_gpu_ptrs(res_gpu)
        dot_batch_nocheck(k_gpu, w_gpu, res_gpu, k_ptrs, w_ptrs, res_ptrs)
        res = res_gpu[0].get()
        single_gpu = la.dot(_k_gpu, _w_gpu)
        print "retry success {}".format(np.allclose(res, k_dot_w))
        print "gpu success {}".format(np.allclose(single_gpu.get(), res))
        assert np.allclose(single_gpu.get(), res)
        raw_input("go?")

def check_update(ctx, b):
    ctx.tps_params[0] = ctx.default_tps_params.copy()
    ctx.update_ptrs()
    xt = ctx.pts_t[0].get()
    p_mat = ctx.proj_mats[b][0].get()
    o_mat = ctx.offset_mats[b][0].get()
    true_res = np.dot(p_mat, xt) + o_mat
    ctx.set_tps_params(ctx.offset_mats[b])
    o_gpu = ctx.tps_params[0].get()
    if not np.allclose(o_gpu, o_mat):
        print "setting tps params failed"
        diff = np.abs(o_mat - o_gpu)
        nz = np.nonzero(diff)
        print nz
        ipy.embed()
        sys.exit(1)
    ctx.update_transform(b)
    p1 = ctx.tps_params[0].get()
    if not np.allclose(true_res, p1):
        print "p1 and true res differ"
        print p1[:3]
        diff = np.abs(p1 - true_res)
        print np.max(diff)
        amax = np.argmax(diff)
        print amax
        nz = np.nonzero(diff)
        print nz[0]
        ipy.embed()
        sys.exit(1)

# @profile
def batch_tps_rpm_bij(src_ctx, tgt_ctx, T_init = 1e-1, T_final = 5e-3, 
                      outlierfrac = 1e-2, outlierprior = 1e-1, outliercutoff = 1e-2, em_iter = EM_ITER_CHEAP):
    """
    computes tps rpm for the clouds in src and tgt in batch
    TODO: Fill out comment cleanly
    """
    ##TODO: add check to ensure that src_ctx and tgt_ctx are formatted properly
    n_iter = len(src_ctx.bend_coefs)
    T_vals = loglinspace(T_init, T_final, n_iter)

    src_ctx.reset_tps_params()
    tgt_ctx.reset_tps_params()
    for i, b in enumerate(src_ctx.bend_coefs):
        T = T_vals[i]
        for _ in range(em_iter):
            src_ctx.transform_points()
            tgt_ctx.transform_points()
            src_ctx.get_target_points(tgt_ctx, outlierprior, outlierfrac, outliercutoff, T)
            src_ctx.update_transform(b)
            # check_update(src_ctx, b)
            tgt_ctx.update_transform(b)
    return src_ctx.bidir_tps_cost(tgt_ctx)

def test_batch_tps_rpm_bij(src_ctx, tgt_ctx, T_init = 1e-1, T_final = 5e-3, 
                           outlierfrac = 1e-2, outlierprior = 1e-1, outliercutoff = .5, em_iter = EM_ITER_CHEAP,
                           test_ind = 0):
    from transformations import ThinPlateSpline, set_ThinPlateSpline
    import tps
    n_iter = len(src_ctx.bend_coefs)
    T_vals = loglinspace(T_init, T_final, n_iter)

    x_nd = src_ctx.pts[test_ind].get()[:src_ctx.dims[test_ind]]
    y_md = tgt_ctx.pts[0].get()[:tgt_ctx.dims[0]]
    (n, d) = x_nd.shape
    (m, _) = y_md.shape

    f = ThinPlateSpline(d)    
    g = ThinPlateSpline(d)    

    src_ctx.reset_tps_params()
    tgt_ctx.reset_tps_params()
    for i, b in enumerate(src_ctx.bend_coefs):
        T = T_vals[i]
        for _ in range(em_iter):
            src_ctx.transform_points()
            tgt_ctx.transform_points()

            xwarped_nd = f.transform_points(x_nd)
            ywarped_md = g.transform_points(y_md)
            gpu_xw = src_ctx.pts_w[test_ind].get()[:n, :]
            gpu_yw = tgt_ctx.pts_w[test_ind].get()[:m, :]
            assert np.allclose(xwarped_nd, gpu_xw, atol=1e-5)
            assert np.allclose(ywarped_md, gpu_yw, atol=1e-5)

            xwarped_nd = gpu_xw
            ywarped_md = gpu_yw

            src_ctx.get_target_points(tgt_ctx, outlierprior, outlierfrac, outliercutoff, T)
            
            fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
            invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
            prob_nm = outlierprior * np.ones((n+1, m+1), np.float32)
            prob_nm[:n, :m] = np.exp( -(fwddist_nm + invdist_nm) / float(2*T))
            prob_nm[n, m] = outlierfrac * np.sqrt(n * m)

            gpu_corr = src_ctx.corr_rm[test_ind].get()
            gpu_corr = gpu_corr.flatten()
            gpu_corr = gpu_corr[:(n + 1) * (m + 1)].reshape(n+1, m+1).astype(np.float32)

            assert np.allclose(prob_nm[:n, :m], gpu_corr[:n, :m], atol=1e-5)
            prob_nm[:n, :m] = gpu_corr[:n, :m]

            r_coefs = np.ones(n+1, np.float32)
            c_coefs = np.ones(m+1, np.float32)
            a_N = np.ones((n+1),dtype = np.float32)
            a_N[n] = m*outlierfrac
            b_M = np.ones((m+1), dtype = np.float32)
            b_M[m] = n*outlierfrac
            for _ in range(DEFAULT_NORM_ITERS):
                r_coefs = a_N/prob_nm.dot(c_coefs)
                rn_c_coefs = c_coefs
                c_coefs = b_M/r_coefs.dot(prob_nm)
            gpu_r_coefs = src_ctx.r_coefs[test_ind].get()[:n+1].reshape(n+1)
            gpu_c_coefs_cn = src_ctx.c_coefs_cn[test_ind].get()[:m+1].reshape(m+1)
            gpu_c_coefs_rn = src_ctx.c_coefs_rn[test_ind].get()[:m+1].reshape(m+1)
            assert np.allclose(r_coefs, gpu_r_coefs, atol=1e-5)
            assert np.allclose(c_coefs, gpu_c_coefs_cn, atol=1e-5)
            assert np.allclose(rn_c_coefs, gpu_c_coefs_rn, atol=1e-5)
            
            prob_nm = prob_nm[:n, :m]
            prob_nm *= gpu_r_coefs[:n, None]
            rn_p_nm = prob_nm * gpu_c_coefs_rn[None, :m]
            cn_p_nm = prob_nm * gpu_c_coefs_cn[None, :m]

            wt_n = rn_p_nm.sum(axis=1)
            gpu_corr_cm = src_ctx.corr_cm[test_ind].get().flatten()[:(n+1)*(m+1)]
            gpu_corr_cm = gpu_corr_cm.reshape(m+1, n+1)## b/c it is column major
            assert np.allclose(wt_n, gpu_corr_cm[m, :n], atol=1e-4)

            
            inlier = wt_n > outliercutoff
            xtarg_nd = np.empty((n, DATA_DIM), np.float32)
            xtarg_nd[inlier, :] = rn_p_nm.dot(y_md)[inlier, :]
            xtarg_nd[~inlier, :] = xwarped_nd[~inlier, :]

            wt_m = cn_p_nm.sum(axis=0)
            assert np.allclose(wt_m, gpu_corr[n, :m], atol=1e-4)

            inlier = wt_m > outliercutoff
            ytarg_md = np.empty((m, DATA_DIM), np.float32)
            ytarg_md[inlier, :] = cn_p_nm.T.dot(x_nd)[inlier, :]
            ytarg_md[~inlier, :] = ywarped_md[~inlier, :]
            
            xt_gpu = src_ctx.pts_t[test_ind].get()[:n, :]
            yt_gpu = tgt_ctx.pts_t[test_ind].get()[:m, :]
            assert np.allclose(xtarg_nd, xt_gpu, atol=1e-4)
            assert np.allclose(ytarg_md, yt_gpu, atol=1e-4)

            src_ctx.update_transform(b)
            tgt_ctx.update_transform(b)

            f_p_mat = src_ctx.proj_mats[b][test_ind].get()[:n+d+1, :n]
            f_o_mat = src_ctx.offset_mats[b][test_ind].get()[:n+d+1]
            b_p_mat = tgt_ctx.proj_mats[b][0].get()[:m+d+1, :m]
            b_o_mat = tgt_ctx.offset_mats[b][0].get()[:m+d+1]
            f_params = f_p_mat.dot(xtarg_nd) + f_o_mat
            g_params = b_p_mat.dot(ytarg_md) + b_o_mat


            gpu_fparams = src_ctx.tps_params[test_ind].get()[:n+d+1]
            gpu_gparams = tgt_ctx.tps_params[test_ind].get()[:m+d+1]
            assert np.allclose(f_params, gpu_fparams, atol=1e-4)
            assert np.allclose(g_params, gpu_gparams, atol=1e-4)


            set_ThinPlateSpline(f, x_nd, gpu_fparams)
            set_ThinPlateSpline(g, y_md, gpu_gparams)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, 1)
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, 1)

    gpu_cost = src_ctx.bidir_tps_cost(tgt_ctx)    
    cpu_cost = f._cost + g._cost
    assert np.isclose(gpu_cost[test_ind], cpu_cost, atol=1e-4)
    
        

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='../data/actions.h5')
    parser.add_argument("--sync", action='store_true')
    parser.add_argument("--n_copies", type=int, default=1)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_full", action='store_true')
    parser.add_argument("--timing_runs", type=int, default=100)
    return parser.parse_args()

if __name__=='__main__':
    reset_cuda()
    args = parse_arguments()
    Globals.sync = args.sync
    src_ctx = SrcContext()
    for _ in range(args.n_copies):
        src_ctx.read_h5(args.input_file)
    f = h5py.File(args.input_file, 'r')    
    tgt_cld = downsample_cloud(f['demo1-seg00']['cloud_xyz'][:])
    f.close()
    tgt_ctx = TgtContext(src_ctx)
    tgt_ctx.set_cld(tgt_cld)
    if args.test_full:
        src_ctx.unit_test(tgt_ctx)
        print "unit tests passed, doing full check on batch tps rpm"
        for i in range(src_ctx.N):
            sys.stdout.write("\rtesting source cloud {}".format(i))
            sys.stdout.flush()
            test_batch_tps_rpm_bij(src_ctx, tgt_ctx, test_ind=i)
        print""
        print "tests succeeded!"
        sys.exit()
    if args.test:
        src_ctx.unit_test(tgt_ctx)
        print "testing batch tps_rps"
        test_batch_tps_rpm_bij(src_ctx, tgt_ctx)
        print "test succeeded!!"
        sys.exit()    
    times = []
    print "batchtps initialized"
    for i in range(args.timing_runs):
        sys.stdout.write("\rRunning Timing test {}/{}".format(i, args.timing_runs))
        sys.stdout.flush()
        start = time.time()
        tgt_ctx.set_cld(scaled_tgt_cld)
        c = batch_tps_rpm_bij(src_ctx, tgt_ctx)
        time_taken = time.time() - start
        times.append(time_taken)
    print "\nTiming Tests Complete"
    print "Batch Size:\t\t\t", src_ctx.N
    print "Mean Compute Time per Batch:\t", np.mean(times)
    print "BiDirectional TPS fits/second:\t", float(args.timing_runs * src_ctx.N) / np.sum(times)
