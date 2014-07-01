import h5py
import numpy as np
import scipy.linalg
import scipy.spatial.distance as ssd
import argparse

from tps import tps_kernel_matrix 
from transformations import unit_boxify
from rapprentice import clouds, plotting_plt

from culinalg_exts import dot_batch, get_gpu_ptrs, dot_batch_nocheck, m_dot_batch

import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray
import scikits.cuda.linalg
from scikits.cuda.linalg import dot as cu_dot
from scikits.cuda.linalg import pinv as cu_pinv
from defaults import N_ITER_CHEAP, DEFAULT_LAMBDA, DS_SIZE, BEND_COEFF_DIGITS
import sys

import IPython as ipy

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str)
    parser.add_argument('--bend_coef_init', type=float, default=DEFAULT_LAMBDA[0])
    parser.add_argument('--bend_coef_final', type=float, default=DEFAULT_LAMBDA[1])
    parser.add_argument('--n_iter', type=int, default=N_ITER_CHEAP)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--cloud_name', type=str, default='cloud_xyz')
    return parser.parse_args()
# @profile
def batch_get_sol_params(x_nd, K_nn, bend_coefs, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
    n, d = x_nd.shape

    x_gpu = gpuarray.to_gpu(x_nd)

    H_arr_gpu = []
    for b in bend_coefs:
        cur_offset = np.zeros((1 + d + n, 1 + d + n), np.float32)
        cur_offset[d+1:, d+1:] = b * K_nn
        cur_offset[1:d+1, 1:d+1] = np.diag(rot_coef)
        H_arr_gpu.append(gpuarray.to_gpu(cur_offset))
    H_ptr_gpu = get_gpu_ptrs(H_arr_gpu)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    F = np.zeros((n + d + 1, d), np.float32)
    F[1:d+1, :d] -= np.diag(rot_coef)
    
    Q = np.c_[np.ones((n,1)), x_nd, K_nn].astype(np.float32)
    F = F.astype(np.float32)
    N = N.astype(np.float32)

    Q_gpu     = gpuarray.to_gpu(Q)
    Q_arr_gpu = [Q_gpu for _ in range(len(bend_coefs))]
    Q_ptr_gpu = get_gpu_ptrs(Q_arr_gpu)

    F_gpu     = gpuarray.to_gpu(F)
    F_arr_gpu = [F_gpu for _ in range(len(bend_coefs))]
    F_ptr_gpu = get_gpu_ptrs(F_arr_gpu)

    N_gpu = gpuarray.to_gpu(N)
    N_arr_gpu = [N_gpu for _ in range(len(bend_coefs))]
    N_ptr_gpu = get_gpu_ptrs(N_arr_gpu)
    
    dot_batch_nocheck(Q_arr_gpu, Q_arr_gpu, H_arr_gpu,
                      Q_ptr_gpu, Q_ptr_gpu, H_ptr_gpu,
                      transa = 'T')
    # N'HN
    NHN_arr_gpu, NHN_ptr_gpu = m_dot_batch((N_arr_gpu, N_ptr_gpu, 'T'),
                                           (H_arr_gpu, H_ptr_gpu, 'N'),
                                           (N_arr_gpu, N_ptr_gpu, 'N'))
    iH_arr = []
    for NHN in NHN_arr_gpu:
        iH_arr.append(scipy.linalg.inv(NHN.get()).copy())
    iH_arr_gpu = [gpuarray.to_gpu_async(iH) for iH in iH_arr]
    iH_ptr_gpu = get_gpu_ptrs(iH_arr_gpu)

    proj_mats   = m_dot_batch((N_arr_gpu,  N_ptr_gpu,   'N'),
                              (iH_arr_gpu, iH_ptr_gpu, 'N'),
                              (N_arr_gpu,  N_ptr_gpu,   'T'),
                              (Q_arr_gpu,  Q_ptr_gpu,   'T'))

    offset_mats = m_dot_batch((N_arr_gpu,  N_ptr_gpu,   'N'),
                              (iH_arr_gpu, iH_ptr_gpu, 'N'),
                              (N_arr_gpu,  N_ptr_gpu,   'T'),
                              (F_arr_gpu,  F_ptr_gpu,   'N'))

    return proj_mats, offset_mats

# @profile
def get_sol_params(x_na, K_nn, bend_coef, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
    """
    precomputes the linear operators to solve this linear system. 
    only dependence on data is through the specified targets

    all thats needed is to compute the righthand side and do a forward solve
    """
    n,d = x_na.shape
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    # QWQ = Q.T.dot(WQ)
    H = Q.T.dot(Q)
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T

    # f = -WQ.T.dot(y_ng)
    # f[1:d+1,0:d] -= np.diag(rot_coefs)
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    tmp = N.T.dot(H.dot(N))
    h_inv = scipy.linalg.inv(tmp)
    
    # z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    # x = N.dot(z)


    ## x = N * (N' * H * N)^-1 * N' * Q' * y + N * (N' * H * N)^-1 * N' * diag(rot_coeffs)
    ## x = proj_mat * y + offset_mat
    ## technically we could reduce this (the N * N^-1 cancel), but H is not full rank, 
    ## so it is better to invert N' * H * N, which is square and full rank
    proj_mat = N.dot(h_inv.dot(N.T.dot(Q.T)))
    F = np.zeros(Q.T.dot(x_na).shape)
    F[1:d+1,0:d] -= np.diag(rot_coefs)
    offset_mat = N.dot(h_inv.dot(N.T.dot(F)))

    res_dict = {'proj_mat': proj_mat, 
                'offset_mat' : offset_mat, 
                'h_inv': h_inv, 
                'N' : N, 
                'rot_coefs' : rot_coefs}

    return bend_coef, res_dict


def downsample_cloud(cloud_xyz):
    return clouds.downsample(cloud_xyz, DS_SIZE)

def main():
    args = parse_arguments()

    f = h5py.File(args.datafile, 'r+')
    
    bend_coefs = np.around(loglinspace(args.bend_coef_init, args.bend_coef_final, args.n_iter), 
                           BEND_COEFF_DIGITS)

    for seg_name, seg_info in f.iteritems():
        if 'inv' in seg_info:
            if args.replace:
                del seg_info['inv'] 
                inv_group = seg_info.create_group('inv')
            else:
                inv_group =  seg_info['inv']
        else:
            inv_group = seg_info.create_group('inv')
        ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
        if ds_key in inv_group:
            scaled_x_na = inv_group[ds_key]['scaled_cloud_xyz'][:]
            K_nn = inv_group[ds_key]['scaled_K_nn'][:]
        else:
            ds_g = inv_group.create_group(ds_key)
            x_na = downsample_cloud(seg_info[args.cloud_name][:, :])
            scaled_x_na, scale_params = unit_boxify(x_na)
            K_nn = tps_kernel_matrix(scaled_x_na)
            ds_g['cloud_xyz'] = x_na
            ds_g['scaled_cloud_xyz'] = scaled_x_na
            ds_g['scaling'] = scale_params[0]
            ds_g['scaled_translation'] = scale_params[1]
            ds_g['scaled_K_nn'] = K_nn

        for bend_coef in bend_coefs:
            if str(bend_coef) in inv_group:
                continue
            
            bend_coef_g = inv_group.create_group(str(bend_coef))
            _, res = get_sol_params(scaled_x_na, K_nn, bend_coef)
            for k, v in res.iteritems():
                bend_coef_g[k] = v

        if args.verbose:
            sys.stdout.write('\rprecomputed tps solver for segment {}'.format(seg_name))
            sys.stdout.flush()
    print ""

    f.close()

if __name__=='__main__':
    main()
            
