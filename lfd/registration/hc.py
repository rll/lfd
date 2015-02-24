"""
Implements

Ting Chen, Baba C. Vemuri, Anand Rangarajan, and Stephan J. Eisenschenk. "Group-Wise Point-Set Registration Using a Novel CDF-Based Havrda-Charvat Divergence,"International Journal of Computer Vision, vol. 86, no. 1, pp. 111-124, 2010.
"""
from __future__ import division

import settings
import numpy as np
import scipy.optimize as so
import tps_experimental
from tps_experimental import ThinPlateSpline, multi_tps_to_params, params_to_multi_tps

def analymin(x, y, beta):
    return -np.log(np.exp(-beta * x) + np.exp(-beta * y)) / beta

def gradymin(x, y, beta=None):
    # # approximate grad min
    # if x == y:
    #     grad = 1
    # else:
    #     exp_m_beta_x = np.exp(-beta * x)
    #     grad = exp_m_beta_x / (exp_m_beta_x + np.exp(-beta * y))
    return (x <= y).astype(float)

def int_p2(x_ld, y_md):
    l, _ = x_ld.shape
    m, _ = y_md.shape
    # v = 0
    # for i in range(l):
    #     for j in range(m):
    #         v += min(x_ld[i,0], y_md[j,0]) * min(x_ld[i,1], y_md[j,1]) #* min(x_ld[i,2], y_md[j,2])
    # v /= l * m

    x_ldm = np.tile(x_ld[:,:,None], (1,1,m))
    v = np.sum(np.prod(np.minimum(x_ldm, y_md.T), axis=1)) / (l * m)
    return v

def grad_p2(x_ld, y_md=None):
    """If y_md is passed, it must not be the same as x_ld."""
    if y_md is None:
        y_md = x_ld
        l, d = x_ld.shape
        m, _ = y_md.shape
        # # naive, d == 2
        # gv_ld = np.zeros((l, d))
        # for i in range(l):
        #     for j in range(m):
        #         if i != j:
        #             gv_ld[i,0] += 2 * gradymin(x_ld[i,0], y_md[j,0]) * min(x_ld[i,1], y_md[j,1])
        #             gv_ld[i,1] += 2 * min(x_ld[i,0], y_md[j,0]) * gradymin(x_ld[i,1], y_md[j,1])
        #         else:
        #             gv_ld[i,0] += min(x_ld[i,1], y_md[j,1])
        #             gv_ld[i,1] += min(x_ld[i,0], y_md[j,0])
        # gv_ld /= l * m

        x_ldm = np.tile(x_ld[:,:,None], (1,1,m))
        gradymin_ldm = gradymin(x_ldm, y_md.T)
        min_ldm = np.minimum(x_ldm, y_md.T)
        gv_ld = np.empty((l, d))
        for i in range(d):
            not_i = np.ones(d, dtype=bool)
            not_i[i] = False
            c_nm = 2 * gradymin_ldm[:,i,:]
            np.fill_diagonal(c_nm, 1)
            gv_ld[:,i] = np.sum(c_nm * np.prod(min_ldm[:,not_i,:], axis=1), axis=1)
        gv_ld /= l * m
    else:
        l, d = x_ld.shape
        m, _ = y_md.shape
        # # naive, d == 2
        # gv_ld = np.zeros((l, d))
        # for i in range(l):
        #     for j in range(m):
        #         gv_ld[i,0] += gradymin(x_ld[i,0], y_md[j,0]) * min(x_ld[i,1], y_md[j,1])
        #         gv_ld[i,1] += min(x_ld[i,0], y_md[j,0]) * gradymin(x_ld[i,1], y_md[j,1])
        # gv_ld /= l * m

        x_ldm = np.tile(x_ld[:,:,None], (1,1,m))
        gradymin_ldm = gradymin(x_ldm, y_md.T)
        min_ldm = np.minimum(x_ldm, y_md.T)
        gv_ld = np.empty((l, d))
        for i in range(d):
            not_i = np.ones(d, dtype=bool)
            not_i[i] = False
            gv_ld[:,i] = np.sum(gradymin_ldm[:,i,:] * np.prod(min_ldm[:,not_i,:], axis=1), axis=1)
        gv_ld /= l * m
    return gv_ld

def groupwise_hc2_obj(x_kld):
    k = len(x_kld)
    alpha = 1/k

    energy1 = 0
    energy2 = 0
    for x_ld in x_kld:
        l , _ = x_ld.shape
        energy2 += alpha * int_p2(x_ld, x_ld)
    for x_ld in x_kld:
        for y_md in x_kld:
            energy1 += (alpha**2) * int_p2(x_ld, y_md)
    energy = -energy1 + energy2

    grad_kld = []
    for r, x_ld in enumerate(x_kld):
        l, d = x_ld.shape
        grad_ld = np.zeros((l, d))
        for s, y_md in enumerate(x_kld):
            if r != s:
                # TODO: why 2?
                grad_ld -= 2 * (alpha**2) * grad_p2(x_ld, y_md)
        grad_ld += (alpha - alpha**2) * grad_p2(x_ld)
        grad_kld.append(grad_ld)
    return energy, grad_kld

def translate_to_R_plus(x_kld, region, ret_translation=False):
    min_x_kld = np.min([np.min(x_ld) for x_ld in x_kld])
    translation = -(min_x_kld - region)
    for x_ld in x_kld:
        x_ld += translation
    if ret_translation:
        ret = x_kld, translation
    else:
        ret = x_kld
    return ret

def groupwise_tps_hc2_obj(z_knd, f_k, reg, rot_reg, y_md=None):
    f_k = params_to_multi_tps(z_knd, f_k)
    
    xwarped_kld = []
    for f in f_k:
        xwarped_kld.append(f.transform_points())
    _, d = xwarped_kld[0].shape
    xwarped_kld = translate_to_R_plus(xwarped_kld, np.ones(d))

    if y_md is None:
        gw_hc2_energy, gw_hc2_grad_kld = groupwise_hc2_obj(xwarped_kld)
    else:
        gw_hc2_energy, gw_hc2_grad_kld = groupwise_hc2_obj(xwarped_kld + [y_md])
    energy = gw_hc2_energy
    grad_knd = []
    for f, gw_hc2_grad_ld in zip(f_k, gw_hc2_grad_kld):
        n, d = f.z_ng.shape
        NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
        NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
        energy += np.trace(f.z_ng.T.dot(reg * f.NKN_nn + NRN_nn).dot(f.z_ng)) - 2 * np.trace(f.z_ng.T.dot(NR_nd))
        grad_nd = f.QN_ln.T.dot(gw_hc2_grad_ld)
        grad_nd += 2 * (reg * f.NKN_nn + NRN_nn).dot(f.z_ng) - 2 * NR_nd
        grad_knd.append(grad_nd.reshape(n*d))
    grad_knd = np.concatenate(grad_knd)
    return energy, grad_knd

def groupwise_tps_hc2(x_kld, y_md=None, ctrl_knd=None, f_init_k=None, opt_iter=100, reg=settings.REG[1], rot_reg=settings.ROT_REG, callback=None):
    """
    Specify y_md to perform biased groupwise registration
    """
    if f_init_k is None:
        if ctrl_knd is None:
            ctrl_knd = x_kld
        else:
            if len(ctrl_knd) != len(x_kld):
                raise ValueError("The number of control points in ctrl_knd is different from the number of point sets in x_kld")
        f_k = []
        for x_ld, ctrl_nd in zip(x_kld, ctrl_knd):
            f = ThinPlateSpline(x_ld, ctrl_nd)
            f_k.append(f)
    else:  
        if len(f_init_k) != len(x_kld):
            raise ValueError("The number of ThinPlateSplines in f_init_k is different from the number of point sets in x_kld")
        f_k = f_init_k

    # translate problem by trans_d. At the end, problem needs to be translated back
    if y_md is not None:
        # shift y_md and f by trans_d so that y_md is in R plus for stability
        _, d = y_md.shape
        [y_trans_md], trans_d = translate_to_R_plus([y_md.copy()], np.ones(d), ret_translation=True) # copy() because it is translated in place
    else:
        d = x_kld[0].shape[1]
        trans_d = np.zeros(d)
        y_trans_md = y_md
    for f in f_k:
        f.trans_g += trans_d

    z_knd = multi_tps_to_params(f_k)

    def opt_callback(z_knd):
        params_to_multi_tps(z_knd, f_k)
        for f in f_k:
            f.trans_g -= trans_d
        callback(f_k, y_md)
        for f in f_k:
            f.trans_g += trans_d
        # print groupwise_tps_hc2_obj(z_knd, f_k, reg, rot_reg, y_md=y_trans_md)[0]

    res = so.fmin_l_bfgs_b(groupwise_tps_hc2_obj, z_knd, None, args=(f_k, reg, rot_reg, y_trans_md), maxfun=opt_iter, callback=opt_callback if callback is not None else None)
    z_knd = res[0]
    
    f_k = params_to_multi_tps(z_knd, f_k)
    for f in f_k:
        f.trans_g -= trans_d
    return f_k

def groupwise_tps_hc2_cov_obj(z_knd, f_k, p_ktd, reg, rot_reg, cov_coef, y_md=None, L_ktkn=None):
    f_k = params_to_multi_tps(z_knd, f_k)
    
    gw_tps_hc2_energy, gw_tps_hc2_grad_knd = groupwise_tps_hc2_obj(z_knd, f_k, reg, rot_reg, y_md=y_md)
    cov_energy, cov_grad_knd = tps_experimental.tps_cov_obj(z_knd, f_k, p_ktd, L_ktkn=L_ktkn)
    energy = gw_tps_hc2_energy + cov_coef * cov_energy
    grad_knd = gw_tps_hc2_grad_knd + cov_coef * cov_grad_knd
    
    return energy, grad_knd

def groupwise_tps_hc2_cov(x_kld, p_ktd, y_md=None, ctrl_knd=None, f_init_k=None, opt_iter=100, reg=settings.REG[1], rot_reg=settings.ROT_REG, cov_coef=settings.COV_COEF, callback=None, multi_callback=None):
    # intitalize z from independent optimizations from the one without covariance
    f_k = groupwise_tps_hc2(x_kld, y_md=y_md, ctrl_knd=ctrl_knd, f_init_k=f_init_k, opt_iter=opt_iter, reg=reg, rot_reg=rot_reg, callback=callback)

    # translate problem by trans_d. At the end, problem needs to be translated back
    if y_md is not None:
        # shift y_md and f by trans_d so that y_md is in R plus for stability
        _, d = y_md.shape
        [y_trans_md], trans_d = translate_to_R_plus([y_md.copy()], np.ones(d), ret_translation=True) # copy() because it is translated in place
    else:
        d = x_kld[0].shape[1]
        trans_d = np.zeros(d)
        y_trans_md = y_md
    for f in f_k:
        f.trans_g += trans_d

    z_knd = multi_tps_to_params(f_k)

    # put together matrix for computing sum of variances
    L_ktkn = tps_experimental.compute_sum_var_matrix(f_k, p_ktd)

    def opt_callback(z_knd):
        params_to_multi_tps(z_knd, f_k)
        for f in f_k:
            f.trans_g -= trans_d
        multi_callback(f_k, p_ktd, y_md)
        for f in f_k:
            f.trans_g += trans_d
        # print groupwise_tps_hc2_cov_obj(z_knd, f_k, p_ktd, reg, rot_reg, cov_coef, y_md=y_trans_md, L_ktkn=L_ktkn)[0]

    res = so.fmin_l_bfgs_b(groupwise_tps_hc2_cov_obj, z_knd, None, args=(f_k, p_ktd, reg, rot_reg, cov_coef, y_trans_md, L_ktkn), maxfun=opt_iter, callback=opt_callback if multi_callback is not None else None)
    z_knd = res[0]
    
    f_k = params_to_multi_tps(z_knd, f_k)
    for f in f_k:
        f.trans_g -= trans_d
    return f_k
