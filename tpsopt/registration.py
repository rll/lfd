"""
Register point clouds to each other


arrays are named like name_abc
abc are subscripts and indicate the what that tensor index refers to

index name conventions:
    m: test point index
    n: training point index
    a: input coordinate
    g: output coordinate
    d: gripper coordinate
"""

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
from rapprentice import svds, math_utils
from transformations import ThinPlateSpline, fit_ThinPlateSpline, set_ThinPlateSpline
import tps
from tps import tps_cost
import IPython as ipy
from constants import BEND_COEF_DIGITS
# from svds import svds


def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

def registration_cost(xyz0, xyz1, f_p_mats=None, f_o_mats=None, b_p_mats=None, b_o_mats=None):
    if f_p_mats is None:        
        f, g = tps_rpm_bij(xyz0, xyz1, n_iter=10)
    else:
        f, g = tps_rpm_bij_presolve(xyz0, xyz1, n_iter=10, f_p_mats=f_p_mats, f_o_mats=f_o_mats,
                                    b_p_mats=b_p_mats, b_o_mats=b_o_mats)
    return f._cost + g._cost

def unit_boxify(x_na):    
    ranges = x_na.ptp(axis=0)
    dlarge = ranges.argmax()
    unscaled_translation = - (x_na.min(axis=0) + x_na.max(axis=0))/2
    scaling = 1./ranges[dlarge]
    scaled_translation = unscaled_translation * scaling
    return x_na*scaling + scaled_translation, (scaling, scaled_translation)    

def unscale_tps(f, src_params, targ_params):
    """Only works in 3d!!"""
    p,q = src_params
    r,s = targ_params
    
    d = len(q)
    
    lin_in = np.eye(d)*p
    trans_in = q
    aff_in = Affine(lin_in, trans_in)
    
    lin_out = np.eye(d)/r
    trans_out = -s/r
    aff_out = Affine(lin_out, trans_out)

    return Composition([aff_in, f, aff_out])
        
# @profile
def tps_rpm_bij(x_nd, y_md, fsolve, gsolve,n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1,
                rad_final = .005, rot_reg = 1e-3, outlierprior=1e-1, outlierfrac=2e-1, vis_cost_xy=None,
                return_corr=False, check_solver=False):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = np.around(loglinspace(reg_init, reg_final, n_iter), BEND_COEF_DIGITS)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g


    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix(prob_nm, 10, outlierprior, outlierfrac)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)                

        fsolve.solve(wt_n, xtarg_nd, regs[i], rot_reg, f)        
        gsolve.solve(wt_m, ytarg_md, regs[i], rot_reg, g)
        if check_solver:
            f_test = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
            g_test = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)
            tol = 1e-4
            assert np.allclose(f.trans_g, f_test.trans_g, atol=tol)
            assert np.allclose(f.lin_ag, f_test.lin_ag, atol=tol)
            assert np.allclose(f.w_ng, f_test.w_ng, atol=tol)
            assert np.allclose(g.trans_g, g_test.trans_g, atol=tol)
            assert np.allclose(g.lin_ag, g_test.lin_ag, atol=tol)
            assert np.allclose(g.w_ng, g_test.w_ng, atol=tol)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    if return_corr:
        return (f, g), corr_nm
    return f,g

def balance_matrix(prob_nm, max_iter, p, outlierfrac, r_N = None):
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = p
    prob_NM[n, :m] = p
    prob_NM[n, m] = p*np.sqrt(n*m)
    
    a_N = np.ones((n+1),'f4')
    a_N[n] = m*outlierfrac
    b_M = np.ones((m+1),'f4')
    b_M[m] = n*outlierfrac
    
    if r_N is None: r_N = np.ones(n+1,'f4')

    for _ in xrange(max_iter):
        c_M = b_M/r_N.dot(prob_NM)
        r_N = a_N/prob_NM.dot(c_M)

    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    
    return prob_NM[:n, :m], r_N, c_M
