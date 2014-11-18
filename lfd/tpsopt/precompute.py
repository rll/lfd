import h5py
import argparse
import sys

import numpy as np
import scipy.linalg
from pycuda import gpuarray
import scikits.cuda.linalg
from scikits.cuda.linalg import pinv as cu_pinv

from settings import GRIPPER_OPEN_CLOSE_THRESH
from tps import tps_kernel_matrix, tps_kernel_matrix2
from lfd.tpsopt.registration import unit_boxify, loglinspace
from lfd.rapprentice import clouds
from culinalg_exts import get_gpu_ptrs, dot_batch_nocheck, m_dot_batch
from settings import N_ITER_CHEAP, DEFAULT_LAMBDA, DS_SIZE, BEND_COEF_DIGITS, EXACT_LAMBDA, N_ITER_EXACT


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str)
    parser.add_argument('--bend_coef_init', type=float, default=DEFAULT_LAMBDA[0])
    parser.add_argument('--bend_coef_final', type=float, default=DEFAULT_LAMBDA[1])
    parser.add_argument('--exact_bend_coef_init', type=float, default=EXACT_LAMBDA[0])
    parser.add_argument('--exact_bend_coef_final', type=float, default=EXACT_LAMBDA[1])
    parser.add_argument('--n_iter', type=int, default=N_ITER_CHEAP)
    parser.add_argument('--exact_n_iter', type=int, default=N_ITER_EXACT)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--cloud_name', type=str, default='cloud_xyz')
    parser.add_argument('--fill_traj', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()
# @profile
def batch_get_sol_params(x_nd, K_nn, bend_coefs, rot_coef):
    n, d = x_nd.shape

    x_gpu = gpuarray.to_gpu(x_nd)

    H_arr_gpu = []
    for b in bend_coefs:
        cur_offset = np.zeros((1 + d + n, 1 + d + n), np.float64)
        cur_offset[d+1:, d+1:] = b * K_nn
        cur_offset[1:d+1, 1:d+1] = np.diag(rot_coef)
        H_arr_gpu.append(gpuarray.to_gpu(cur_offset))
    H_ptr_gpu = get_gpu_ptrs(H_arr_gpu)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    F = np.zeros((n + d + 1, d), np.float64)
    F[1:d+1, :d] += np.diag(rot_coef)
    
    Q = np.c_[np.ones((n,1)), x_nd, K_nn].astype(np.float64)
    F = F.astype(np.float64)
    N = N.astype(np.float64)

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

def test_batch_get_sol_params(f, bend_coefs, rot_coef, atol=1e-7, index=0):
    seg_info = f.items()[index][1]
    inv_group =  seg_info['inv']
    ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
    x_nd = inv_group[ds_key]['scaled_cloud_xyz'][:]
    K_nn = inv_group[ds_key]['scaled_K_nn'][:]

    n, d = x_nd.shape

    x_gpu = gpuarray.to_gpu(x_nd)

    H_arr_gpu = []
    for b in bend_coefs:
        cur_offset = np.zeros((1 + d + n, 1 + d + n), np.float64)
        cur_offset[d+1:, d+1:] = b * K_nn
        cur_offset[1:d+1, 1:d+1] = np.diag(rot_coef)
        H_arr_gpu.append(gpuarray.to_gpu(cur_offset))
    H_ptr_gpu = get_gpu_ptrs(H_arr_gpu)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_nd]].T
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    F = np.zeros((n + d + 1, d), np.float64)
    F[1:d+1, :d] += np.diag(rot_coef)
    
    Q = np.c_[np.ones((n,1)), x_nd, K_nn].astype(np.float64)
    F = F.astype(np.float64)
    N = N.astype(np.float64)

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
    QTQ = Q.T.dot(Q)
    H_list = []
    for i, bend_coef in enumerate(bend_coefs):
        H = QTQ
        H[d+1:,d+1:] += bend_coef * K_nn
        rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
        H[1:d+1, 1:d+1] += np.diag(rot_coefs)
        # ipdb.set_trace()
        H_list.append(H)

    # N'HN
    NHN_arr_gpu, NHN_ptr_gpu = m_dot_batch((N_arr_gpu, N_ptr_gpu, 'T'),
                                           (H_arr_gpu, H_ptr_gpu, 'N'),
                                           (N_arr_gpu, N_ptr_gpu, 'N'))

    NHN_list = [N.T.dot(H.dot(N)) for H in H_list]
    for i, NHN in enumerate(NHN_list):
        assert(np.allclose(NHN, NHN_arr_gpu[i].get(), atol=atol))

    iH_arr = []
    for NHN in NHN_arr_gpu:
        iH_arr.append(scipy.linalg.inv(NHN.get()).copy())

    h_inv_list = [scipy.linalg.inv(NHN) for NHN in NHN_list]
    assert(np.allclose(iH_arr, h_inv_list, atol=atol))

    iH_arr_gpu = [gpuarray.to_gpu_async(iH) for iH in iH_arr]
    iH_ptr_gpu = get_gpu_ptrs(iH_arr_gpu)

    proj_mats   = m_dot_batch((N_arr_gpu,  N_ptr_gpu,   'N'),
                              (iH_arr_gpu, iH_ptr_gpu, 'N'),
                              (N_arr_gpu,  N_ptr_gpu,   'T'),
                              (Q_arr_gpu,  Q_ptr_gpu,   'T'))
    
    proj_mats_list = [N.dot(h_inv.dot(N.T.dot(Q.T))) for h_inv in h_inv_list]
    assert(np.allclose(proj_mats_list, proj_mats[0][index].get(), atol=atol))

    offset_mats = m_dot_batch((N_arr_gpu,  N_ptr_gpu,   'N'),
                              (iH_arr_gpu, iH_ptr_gpu, 'N'),
                              (N_arr_gpu,  N_ptr_gpu,   'T'),
                              (F_arr_gpu,  F_ptr_gpu,   'N'))
    offset_mats_list = [N.dot(h_inv.dot(N.T.dot(F))) for h_inv in h_inv_list]
    assert(np.allclose(offset_mats_list, offset_mats[0][index].get(), atol=atol))

def get_exact_solver(x_na, K_nn, bend_coefs, rot_coef):
    """
    precomputes several of the matrix products needed to fit a TPS w/o the approximations
    for the batch computation

    a TPS is fit by solving the system
    N'(Q'WQ +O_b)N z = -N'(Q'W'y - N'R)
    x = Nz

    This function returns a tuple
    N, QN, N'O_bN, N'R
    where N'O_bN is a dict mapping the desired bending coefs to the appropriate product
    """
    n,d = x_na.shape
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
    for b in bend_coefs:
        O = np.zeros((n+d+1, n+d+1))
        O[d+1:, d+1:] += b * K_nn
        O[1:d+1, 1:d+1] += np.diag(rot_coef)
        NON[b] = N.T.dot(O.dot(N))
    return N, QN, NON, NR


# @profile
def get_sol_params(x_na, K_nn, bend_coef, rot_coef):
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
    F[1:d+1,0:d] += np.diag(rot_coefs)
    offset_mat = N.dot(h_inv.dot(N.T.dot(F)))

    res_dict = {'proj_mat': proj_mat, 
                'offset_mat' : offset_mat, 
                'h_inv': h_inv, 
                'N' : N, 
                'rot_coefs' : rot_coefs}

    return bend_coef, res_dict

# Copied from lfd/sim_util.py:
def get_opening_closing_inds(finger_traj):
    
    mult = 5.0
    GRIPPER_L_FINGER_OPEN_CLOSE_THRESH = mult * GRIPPER_OPEN_CLOSE_THRESH

    # indices BEFORE transition occurs
    opening_inds = np.flatnonzero((finger_traj[1:] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    closing_inds = np.flatnonzero((finger_traj[1:] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    
    return opening_inds, closing_inds

def gripper_joint2gripper_l_finger_joint_values(gripper_joint_vals):
    """
    Only the %s_gripper_l_finger_joint%lr can be controlled (this is the joint returned by robot.GetManipulator({"l":"leftarm", "r":"rightarm"}[lr]).GetGripperIndices())
    The rest of the gripper joints (like %s_gripper_joint%lr) are mimiced and cannot be controlled directly
    """
    mult = 5.0
    gripper_l_finger_joint_vals = mult * gripper_joint_vals
    return gripper_l_finger_joint_vals



def downsample_cloud(cloud_xyz):
    return clouds.downsample(cloud_xyz, DS_SIZE)

def main():
    scikits.cuda.linalg.init()
    args = parse_arguments()

    f = h5py.File(args.datafile, 'r+')
    
    bend_coefs = np.around(loglinspace(args.bend_coef_init, args.bend_coef_final, args.n_iter), 
                           BEND_COEF_DIGITS)

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
            x_na = downsample_cloud(seg_info[args.cloud_name][:, :3])
            scaled_x_na, scale_params = unit_boxify(x_na)
            K_nn = tps_kernel_matrix(scaled_x_na)
            if args.fill_traj:
                r_traj          = seg_info['r_gripper_tool_frame']['hmat'][:, :3, 3]
                l_traj          = seg_info['l_gripper_tool_frame']['hmat'][:, :3, 3]
                scaled_r_traj   = r_traj * scale_params[0] + scale_params[1]
                scaled_l_traj   = l_traj * scale_params[0] + scale_params[1]
                scaled_r_traj_K = tps_kernel_matrix2(scaled_r_traj, scaled_x_na)
                scaled_l_traj_K = tps_kernel_matrix2(scaled_l_traj, scaled_x_na)
                ds_g['scaled_r_traj']      = scaled_r_traj
                ds_g['scaled_l_traj']      = scaled_l_traj
                ds_g['scaled_r_traj_K']    = scaled_r_traj_K
                ds_g['scaled_l_traj_K']    = scaled_l_traj_K
                # Precompute l,r closing indices
                lr2finger_traj = {}
                for lr in 'lr':
                    arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
                    lr2finger_traj[lr] = gripper_joint2gripper_l_finger_joint_values(np.asarray(seg_info['%s_gripper_joint'%lr]))[:,None]
                    opening_inds, closing_inds = get_opening_closing_inds(lr2finger_traj[lr])
                    if '%s_closing_inds'%lr in seg_info:
                        del seg_info['%s_closing_inds'%lr]
                    if not closing_inds:
                        closing_inds = False
                    seg_info['%s_closing_inds'%lr] = closing_inds
#
            ds_g['cloud_xyz']          = x_na
            ds_g['scaled_cloud_xyz']   = scaled_x_na
            ds_g['scaling']            = scale_params[0]
            ds_g['scaled_translation'] = scale_params[1]
            ds_g['scaled_K_nn']        = K_nn


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

    if args.test:
        atol = 1e-7
        print 'Running batch get sol params test with atol = ', atol
        test_batch_get_sol_params(f, [bend_coef], atol=atol)
        print 'batch sol params test succeeded'
    print ""

    bend_coefs = np.around(loglinspace(args.exact_bend_coef_init, args.exact_bend_coef_final,
                                       args.exact_n_iter), 
                           BEND_COEF_DIGITS)
    for seg_name, seg_info in f.iteritems():
        if 'solver' in seg_info:
            if args.replace:
                del seg_info['solver']
                solver_g = seg_info.create_group('solver')
            else:
                solver_g = seg_info['solver']
        else:
            solver_g = seg_info.create_group('solver')
        x_nd = seg_info['inv'][ds_key]['scaled_cloud_xyz'][:]
        K_nn = seg_info['inv'][ds_key]['scaled_K_nn'][:]
        N, QN, NON, NR = get_exact_solver(x_nd, K_nn, bend_coefs)
        solver_g['N']    = N
        solver_g['QN']   = QN
        solver_g['NR']   = NR
        solver_g['x_nd'] = x_nd
        solver_g['K_nn'] = K_nn
        NON_g = solver_g.create_group('NON')
        for b in bend_coefs:
            NON_g[str(b)] = NON[b]
        if args.verbose:
            sys.stdout.write('\rprecomputed exact tps solver for segment {}'.format(seg_name))
            sys.stdout.flush()
    print ""


    f.close()

if __name__=='__main__':
    main()
            
