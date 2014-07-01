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


def fit_ThinPlateSpline_RotReg(x_na, y_ng, bend_coef = .1, rot_coefs = (0.01,0.01,0.0025),scale_coef=.01):
    import fastrapp
    f = ThinPlateSpline()
    rfunc = fastrapp.rot_reg
    fastrapp.set_coeffs(rot_coefs, scale_coef)
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit_regrot(x_na, y_ng, bend_coef, rfunc)
    f.x_na = x_na
    return f            


def unit_boxify(x_na):    
    ranges = x_na.ptp(axis=0)
    dlarge = ranges.argmax()
    unscaled_translation = - (x_na.min(axis=0) + x_na.max(axis=0))/2
    scaling = 1./ranges[dlarge]
    scaled_translation = unscaled_translation * scaling
    return x_na*scaling + scaled_translation, (scaling, scaled_translation)
    
def unscale_tps_3d(f, src_params, targ_params):
    """Only works in 3d!!"""
    assert len(f.trans_g) == 3
    p,q = src_params
    r,s = targ_params
    print p,q,r,s
    fnew = ThinPlateSpline()
    fnew.x_na = (f.x_na  - q[None,:])/p 
    fnew.w_ng = f.w_ng * p / r
    fnew.lin_ag = f.lin_ag * p / r
    fnew.trans_g = (f.trans_g  + f.lin_ag.T.dot(q) - s)/r
    
    return fnew

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
    
    

def tps_rpm(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg=1e-4,
            plotting = False, f_init = None, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        # f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.1, max_iter=10)

        wt_n = corr_nm.sum(axis=1)


        targ_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_nd, corr_nm, wt_n, f)
        
        
        f = fit_ThinPlateSpline(x_nd, targ_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)

    return f

def tps_rpm_bootstrap(x_nd, y_md, z_kd, xy_corr, n_init_iter = 10, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, old_xyz=None, new_xyz=None):
    """
    modification for tps_rpm in order to use bootstrapping info

    return a warp taking x to z, proceeds by warping y->z, then intializes tps_rpm with the correspondences from that warping
    """
    _, _, yz_corr = tps_rpm_bij(y_md, z_kd, n_iter = n_iter, reg_init = reg_init, reg_final = reg_final, 
                                rad_init = rad_init, rad_final = rad_final, rot_reg = rot_reg, plotting=plotting,
                                plot_cb = plot_cb, old_xyz = old_xyz, new_xyz = new_xyz, return_corr = True)
    xz_corr = xy_corr.dot(yz_corr)
    # corr_nk, r_N, _ =  balance_matrix3(xz_corr, 10, 1e-1, 1e-2)
    # corr_nk += 1e-9
    corr_nk = xz_corr
        
    wt_n = corr_nk.sum(axis=1)
    wt_k = corr_nk.sum(axis=0)

    xtarg_nd = (corr_nk/wt_n[:,None]).dot(z_kd)
    ztarg_kd = (corr_nk/wt_k[None,:]).T.dot(x_nd)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = reg_final, wt_n=wt_n, rot_coef = rot_reg)
    g = fit_ThinPlateSpline(z_kd, ztarg_kd, bend_coef = reg_final, wt_n=wt_k, rot_coef = rot_reg)
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, reg_final, wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ztarg_kd, reg_final, wt_n=wt_k)/wt_k.mean()
    print 'cost:\t', f._cost + g._cost
    return (f, g, corr_nk)


def tps_rpm_bij(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, 
                rot_reg = 1e-3):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if not f_init:
        f = ThinPlateSpline(d)
        f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    else:
        f = f_init
    if not f_init:
        g = ThinPlateSpline(d)
        g.trans_g = -f.trans_g
    else:
        g = g_init


    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 1e-2)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        inlier = wt_n > 1e-2
        xtarg_nd = np.empty(x_nd.shape)
        xtarg_nd[inlier, :] = (corr_nm/wt_n[:,None]).dot(y_md)[inlier, :]
        xtarg_nd[~inlier, :] = xwarped_nd[~inlier, :] 

        inlier = wt_m > 1e-2
        ytarg_md = np.empty(y_md.shape)
        ytarg_md[inlier, :] = (corr_nm/wt_m[:,None]).dot(y_md)[inlier, :]
        ytarg_md[~inlier, :] = ywarped_md[~inlier, :] 
                
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    if return_corr:
        return f, g, corr_nm
    return f,g
def tps_rpm_bij_presolve(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, 
                         rot_reg = 1e-3, f_p_mats = None, f_o_mats = None, b_p_mats = None, b_o_mats = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    """
    assert f_p_mats != None
    assert f_o_mats != None
    assert b_p_mats != None
    assert b_o_mats != None
    _,d=x_nd.shape
    regs = np.around(loglinspace(reg_init, reg_final, n_iter), 6)
    rads = loglinspace(rad_init, rad_final, n_iter)
    f = ThinPlateSpline(d)    
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 1e-2)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        inlier = wt_n > 1e-2
        xtarg_nd = np.empty(x_nd.shape)
        xtarg_nd[inlier, :] = (corr_nm/wt_n[:,None]).dot(y_md)[inlier, :]
        xtarg_nd[~inlier, :] = xwarped_nd[~inlier, :] 

        inlier = wt_m > 1e-2
        ytarg_md = np.empty(y_md.shape)
        ytarg_md[inlier, :] = x_nd.dot(corr_nm/wt_m[:,None])[inlier, :]
        ytarg_md[~inlier, :] = ywarped_md[~inlier, :] 
        
        f_params = f_p_mats[regs[i]].dot(xtarg_nd) + f_o_mats[regs[i]]
        set_ThinPlateSpline(f, x_nd, f_params)
        g_params = b_p_mats[regs[i]].dot(ytarg_md) + b_o_mats[regs[i]]
        set_ThinPlateSpline(g, y_md, g_params)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i])
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i])
    return f,g

def tps_reg_cost(f):
    K_nn = tps.tps_kernel_matrix(f.x_na)
    cost = 0
    for w in f.w_ng.T:
        cost += w.dot(K_nn.dot(w))
    return cost
    
def logmap(m):
    "http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29"
    theta = np.arccos(np.clip((np.trace(m) - 1)/2,-1,1))
    return (1/(2*np.sin(theta))) * np.array([[m[2,1] - m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]]), theta


def balance_matrix3(prob_nm, max_iter, p, outlierfrac, r_N = None):
    
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

def balance_matrix(prob_nm, p, max_iter=20, ratio_err_tol=1e-3):
    n,m = prob_nm.shape
    pnoverm = (float(p)*float(n)/float(m))
    for _ in xrange(max_iter):
        colsums = pnoverm + prob_nm.sum(axis=0)        
        prob_nm /=  + colsums[None,:]
        rowsums = p + prob_nm.sum(axis=1)
        prob_nm /= rowsums[:,None]
        
        if ((rowsums-1).__abs__() < ratio_err_tol).all() and ((colsums-1).__abs__() < ratio_err_tol).all():
            break


    return prob_nm

def calc_correspondence_matrix(x_nd, y_md, r, p, max_iter=20):
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
    
    
    prob_nm = np.exp(-dist_nm / r)
    # Seems to work better without **2
    # return balance_matrix(prob_nm, p=p, max_iter = max_iter, ratio_err_tol = ratio_err_tol)
    outlierfrac = 1e-1
    return balance_matrix3(prob_nm, max_iter, p, outlierfrac)


def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x


def fit_score(src, targ, dist_param):
    "how good of a partial match is src to targ"
    sqdists = ssd.cdist(src, targ,'sqeuclidean')
    return -np.exp(-sqdists/dist_param**2).sum()

def orthogonalize3_cross(mats_n33):
    "turns each matrix into a rotation"

    x_n3 = mats_n33[:,:,0]
    # y_n3 = mats_n33[:,:,1]
    z_n3 = mats_n33[:,:,2]

    znew_n3 = math_utils.normr(z_n3)
    ynew_n3 = math_utils.normr(np.cross(znew_n3, x_n3))
    xnew_n3 = math_utils.normr(np.cross(ynew_n3, znew_n3))

    return np.concatenate([xnew_n3[:,:,None], ynew_n3[:,:,None], znew_n3[:,:,None]],2)

def orthogonalize3_svd(x_k33):
    u_k33, _s_k3, v_k33 = svds.svds(x_k33)
    return (u_k33[:,:,:,None] * v_k33[:,None,:,:]).sum(axis=2)

def orthogonalize3_qr(_x_k33):
    raise NotImplementedError
