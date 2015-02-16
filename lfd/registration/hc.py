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

def int_p2(x_nd, y_md):
    n, _ = x_nd.shape
    m, _ = y_md.shape
    # v = 0
    # for i in range(n):
    #     for j in range(m):
    #         v += min(x_nd[i,0], y_md[j,0]) * min(x_nd[i,1], y_md[j,1]) #* min(x_nd[i,2], y_md[j,2])
    # v /= n * m

    x_ndm = np.tile(x_nd[:,:,None], (1,1,m))
    v = np.sum(np.prod(np.minimum(x_ndm, y_md.T), axis=1)) / (n * m)
    return v

def grad_p2(x_nd, y_md=None):
    """If y_md is passed, it must not be the same as x_nd."""
    if y_md is None:
        y_md = x_nd
        n, d = x_nd.shape
        m, _ = y_md.shape
        # # naive, d == 2
        # gv_nd = np.zeros((n, d))
        # for i in range(n):
        #     for j in range(m):
        #         if i != j:
        #             gv_nd[i,0] += 2 * gradymin(x_nd[i,0], y_md[j,0]) * min(x_nd[i,1], y_md[j,1])
        #             gv_nd[i,1] += 2 * min(x_nd[i,0], y_md[j,0]) * gradymin(x_nd[i,1], y_md[j,1])
        #         else:
        #             gv_nd[i,0] += min(x_nd[i,1], y_md[j,1])
        #             gv_nd[i,1] += min(x_nd[i,0], y_md[j,0])
        # gv_nd /= n * m

        x_ndm = np.tile(x_nd[:,:,None], (1,1,m))
        gradymin_ndm = gradymin(x_ndm, y_md.T)
        min_ndm = np.minimum(x_ndm, y_md.T)
        gv_nd = np.empty((n, d))
        for i in range(d):
            not_i = np.ones(d, dtype=bool)
            not_i[i] = False
            c_nm = 2 * gradymin_ndm[:,i,:]
            np.fill_diagonal(c_nm, 1)
            gv_nd[:,i] = np.sum(c_nm * np.prod(min_ndm[:,not_i,:], axis=1), axis=1)
        gv_nd /= n * m
    else:
        n, d = x_nd.shape
        m, _ = y_md.shape
        # # naive, d == 2
        # gv_nd = np.zeros((n, d))
        # for i in range(n):
        #     for j in range(m):
        #         gv_nd[i,0] += gradymin(x_nd[i,0], y_md[j,0]) * min(x_nd[i,1], y_md[j,1])
        #         gv_nd[i,1] += min(x_nd[i,0], y_md[j,0]) * gradymin(x_nd[i,1], y_md[j,1])
        # gv_nd /= n * m

        x_ndm = np.tile(x_nd[:,:,None], (1,1,m))
        gradymin_ndm = gradymin(x_ndm, y_md.T)
        min_ndm = np.minimum(x_ndm, y_md.T)
        gv_nd = np.empty((n, d))
        for i in range(d):
            not_i = np.ones(d, dtype=bool)
            not_i[i] = False
            gv_nd[:,i] = np.sum(gradymin_ndm[:,i,:] * np.prod(min_ndm[:,not_i,:], axis=1), axis=1)
        gv_nd /= n * m
    return gv_nd

def hc2_obj(x_knd):
    k = len(x_knd)
    alpha = 1/k

    energy1 = 0
    energy2 = 0
    for x_nd in x_knd:
        n , _ = x_nd.shape
        energy2 += alpha * int_p2(x_nd, x_nd)
    for x_nd in x_knd:
        for y_md in x_knd:
            energy1 += (alpha**2) * int_p2(x_nd, y_md)
    energy = -energy1 + energy2

    grads = []
    for l, x_nd in enumerate(x_knd):
        n, d = x_nd.shape
        grad_tmp = np.zeros((n, d))
        for s, y_md in enumerate(x_knd):
            if l != s:
                # TODO: why 2?
                grad_tmp += 2 * (alpha**2) * grad_p2(x_nd, y_md)
        grad = -grad_tmp + (alpha - alpha**2) * grad_p2(x_nd)
        grads.append(grad)
    return energy, grads

def translate_to_R_plus(x_knd, region):
    min_x_knd = np.min([np.min(x_nd) for x_nd in x_knd])
    for x_nd in x_knd:
        x_nd -= min_x_knd - region
    return x_knd

def tps_hc2_obj(z_knd, f_k, reg, rot_reg):
    xwarped_knd = []
    i = 0
    for f in f_k:
        n, d = f.z_ng.shape
        f.z_ng = z_knd[i*d:(i+n)*d]
        xwarped_knd.append(f.transform_points())
        i += n
    xwarped_knd = translate_to_R_plus(xwarped_knd, np.ones(d))

    hc2_energy, hc2_grad_knd = hc2_obj(xwarped_knd)
    energy = hc2_energy
    grad_knd = []
    for f, hc2_grad_nd in zip(f_k, hc2_grad_knd):
        n, d = f.z_ng.shape
        NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
        NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
        energy += np.trace(f.z_ng.T.dot(reg * f.NKN_nn + NRN_nn).dot(f.z_ng)) - 2 * np.trace(f.z_ng.T.dot(NR_nd))
        grad_nd = f.QN_ln.T.dot(hc2_grad_nd)
        grad_nd += 2 * (reg * f.NKN_nn + NRN_nn).dot(f.z_ng) - 2 * NR_nd
        grad_knd.append(grad_nd.reshape(n*d))
    grad_knd = np.concatenate(grad_knd)
    return energy, grad_knd

def tps_hc2(x_knd, opt_iter=100, reg=settings.REG[1], rot_reg=settings.ROT_REG, callback=None):
    f_k = []
    z_knd = []
    for x_nd in x_knd:
        n, d = x_nd.shape
        f = ThinPlateSpline(x_nd, x_nd)
        f_k.append(f)
        z_knd.append(f.z_ng.reshape(n*d))
    z_knd = np.concatenate(z_knd)

    res = so.fmin_l_bfgs_b(tps_hc2_obj, z_knd, None, args=(f_k, reg, rot_reg), maxfun=opt_iter, callback=callback)
    z_knd = res[0]
    
    i = 0
    for f in f_k:
        n, d = f.z_ng.shape
        f.z_ng = z_knd[i*d:(i+n)*d]
        i += n

    return f_k
