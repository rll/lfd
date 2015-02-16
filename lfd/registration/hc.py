"""
Implements

Ting Chen, Baba C. Vemuri, Anand Rangarajan, and Stephan J. Eisenschenk. "Group-Wise Point-Set Registration Using a Novel CDF-Based Havrda-Charvat Divergence,"International Journal of Computer Vision, vol. 86, no. 1, pp. 111-124, 2010.
"""
from __future__ import division

import settings
import numpy as np
import scipy.optimize as so
from tps_experimental import ThinPlateSpline

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

def hc2_obj(x_kld):
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

def translate_to_R_plus(x_kld, region):
    min_x_kld = np.min([np.min(x_ld) for x_ld in x_kld])
    for x_ld in x_kld:
        x_ld -= min_x_kld - region
    return x_kld

def tps_hc2_obj(z_knd, f_k, reg, rot_reg):
    xwarped_kld = []
    i = 0
    for f in f_k:
        n, d = f.z_ng.shape
        f.z_ng = z_knd[i*d:(i+n)*d]
        xwarped_kld.append(f.transform_points())
        i += n
    xwarped_kld = translate_to_R_plus(xwarped_kld, np.ones(d))

    hc2_energy, hc2_grad_kld = hc2_obj(xwarped_kld)
    energy = hc2_energy
    grad_knd = []
    for f, hc2_grad_ld in zip(f_k, hc2_grad_kld):
        n, d = f.z_ng.shape
        NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
        NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
        energy += np.trace(f.z_ng.T.dot(reg * f.NKN_nn + NRN_nn).dot(f.z_ng)) - 2 * np.trace(f.z_ng.T.dot(NR_nd))
        grad_nd = f.QN_ln.T.dot(hc2_grad_ld)
        grad_nd += 2 * (reg * f.NKN_nn + NRN_nn).dot(f.z_ng) - 2 * NR_nd
        grad_knd.append(grad_nd.reshape(n*d))
    grad_knd = np.concatenate(grad_knd)
    return energy, grad_knd

def tps_hc2(x_kld, ctrl_knd=None, opt_iter=100, reg=settings.REG[1], rot_reg=settings.ROT_REG, callback=None):
    if ctrl_knd is None:
        ctrl_knd = x_kld

    f_k = []
    z_knd = []
    for x_ld, ctrl_nd in zip(x_kld, ctrl_knd):
        n, d = ctrl_nd.shape
        f = ThinPlateSpline(x_ld, ctrl_nd)
        f_k.append(f)
        z_knd.append(f.z_ng.reshape(n*d))
    z_knd = np.concatenate(z_knd)

    def opt_callback(z_knd):
        i = 0
        for f in f_k:
            n, d = f.z_ng.shape
            f.z_ng = z_knd[i*d:(i+n)*d]
            i += n
        callback(f_k)

    res = so.fmin_l_bfgs_b(tps_hc2_obj, z_knd, None, args=(f_k, reg, rot_reg), maxfun=opt_iter, callback=opt_callback)
    z_knd = res[0]
    
    i = 0
    for f in f_k:
        n, d = f.z_ng.shape
        f.z_ng = z_knd[i*d:(i+n)*d]
        i += n

    return f_k
