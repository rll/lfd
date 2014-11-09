"""
Functions for fitting and applying thin plate spline transformations
"""
from __future__ import division

from constants import TpsConstant as tpsc
import numpy as np
import scipy.spatial.distance as ssd
from transformation import Transformation

def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x

def tps_apply_kernel(distmat, dim):
    """
    if d=2: 
        k(r) = 4 * r^2 log(r)
       d=3:
        k(r) = -r
            
    import numpy as np, scipy.spatial.distance as ssd
    x = np.random.rand(100,2)
    d=ssd.squareform(ssd.pdist(x))
    print np.clip(np.linalg.eigvalsh( 4 * d**2 * log(d+1e-9) ),0,inf).mean()
    print np.clip(np.linalg.eigvalsh(-d),0,inf).mean()
    
    Note the actual coefficients (from http://www.geometrictools.com/Documentation/ThinPlateSplines.pdf)
    d=2: 1/(8*sqrt(pi)) = 0.070523697943469535
    d=3: gamma(-.5)/(16*pi**1.5) = -0.039284682964880184
    """

    if dim==2:       
        return 4 * distmat**2 * np.log(distmat+1e-20)
        
    elif dim ==3:
        return -distmat
    else:
        raise NotImplementedError
    
    
def tps_kernel_matrix(x_na):
    dim = x_na.shape[1]
    distmat = ssd.squareform(ssd.pdist(x_na))
    return tps_apply_kernel(distmat,dim)

def tps_kernel_matrix2(x_na, y_ma):
    dim = x_na.shape[1]
    distmat = ssd.cdist(x_na, y_ma)
    return tps_apply_kernel(distmat, dim)

def tps_eval(x_ma, lin_ag, trans_g, w_ng, x_na):
    K_mn = tps_kernel_matrix2(x_ma, x_na)
    return np.dot(K_mn, w_ng) + np.dot(x_ma, lin_ag) + trans_g[None,:]

def tps_grad(x_ma, lin_ag, _trans_g, w_ng, x_na):
    _N, D = x_na.shape
    M = x_ma.shape[0]

    assert x_ma.shape[1] == 3
    dist_mn = ssd.cdist(x_ma, x_na,'euclidean')

    grad_mga = np.empty((M,D,D))

    lin_ga = lin_ag.T
    for a in xrange(D):
        diffa_mn = x_ma[:,a][:,None] - x_na[:,a][None,:]
        grad_mga[:,:,a] = lin_ga[None,:,a] - np.dot(nan2zero(diffa_mn/dist_mn),w_ng)
    return grad_mga

def solve_eqp1(H, f, A):
    """solve equality-constrained qp
    min .5 tr(x'Hx) + tr(f'x)
    s.t. Ax = 0
    """    
    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars
    assert A.shape[1] == n_vars
    n_cnts = A.shape[0]
    
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    # columns of N span the null space
    
    # x = Nz
    # then problem becomes unconstrained minimization .5 z'N'HNz + z'N'f
    # N'HNz + N'f = 0
    L = N.T.dot(H.dot(N))
    R = -N.T.dot(f)
    z = np.linalg.solve(L, R)
    x = N.dot(z)
    
    return x

def tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape
    
    K_nn = tps_kernel_matrix(x_na)
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
    
    solve_dim_separately = not np.isscalar(bend_coef) or (wt_n.ndim > 1 and wt_n.shape[1] > 1)
    
    if not solve_dim_separately:
        WQ = wt_n[:,None] * Q
        QWQ = Q.T.dot(WQ)
        H = QWQ
        H[d+1:,d+1:] += bend_coef * K_nn
        H[1:d+1, 1:d+1] += np.diag(rot_coefs)
        
        f = -WQ.T.dot(y_ng)
        f[1:d+1,0:d] -= np.diag(rot_coefs)
        
        Theta = solve_eqp1(H,f,A)
    else:
        bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
        if wt_n.ndim == 1:
            wt_n = wt_n[:,None]
        if wt_n.shape[1] == 1:
            wt_n = np.tile(wt_n, (1,d))
        Theta = np.empty((1+d+n,d))
        for i in range(d):
            WQ = wt_n[:,i][:,None] * Q
            QWQ = Q.T.dot(WQ)
            H = QWQ
            H[d+1:,d+1:] += bend_coefs[i] * K_nn
            H[1:d+1, 1:d+1] += np.diag(rot_coefs)
             
            f = -WQ.T.dot(y_ng[:,i])
            f[1+i] -= rot_coefs[i]
             
            Theta[:,i] = solve_eqp1(H,f,A)
                                                            
    return Theta[1:d+1], Theta[0], Theta[d+1:]

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
        
        self.y_ng = np.zeros((0,d))
        self.bend_coef = 0
        self.rot_coef = 0
        self.wt_n = np.zeros(0)
    
    @staticmethod
    def fit_ThinPlateSpline(x_na, y_ng, bend_coef, rot_coef, wt_n):
        """
        x_na: source cloud
        y_ng: target cloud
        smoothing: penalize non-affine part
        angular_spring: penalize rotation
        wt_n: weight the points
        
        Solves the optimization problem
        min \sum{i=1}^n wt_n_i ||y_ng_i - f(x_na_i)||_2^2 + bend_coef tr(w_ng' K_nn w_ng) + tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
        s.t. x_na' w_ng = 0
             1' w_ng = 0
        """
        f = ThinPlateSpline()
        f.set_ThinPlateSpline(x_na, y_ng, bend_coef, rot_coef, wt_n)
        return f

    def set_ThinPlateSpline(self, x_na, y_ng, bend_coef, rot_coef, wt_n, theta=None):
        if theta is None:
            self.lin_ag, self.trans_g, self.w_ng = tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
        else:
            d = x_na.shape[1]
            self.trans_g = theta[0]
            self.lin_ag  = theta[1:d+1]
            self.w_ng    = theta[d+1:]
        self.x_na = x_na
        self.y_ng = y_ng
        self.bend_coef = bend_coef
        self.rot_coef = rot_coef
        self.wt_n = wt_n
    
    def transform_points(self, x_ma):
        y_ng = tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return y_ng

    def compute_jacobian(self, x_ma):
        grad_mga = tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
    
    def get_objective(self):
        """
        Returns the following 3 objectives
        \sum{i=1}^n wt_n_i ||y_ng_i - f(x_na_i)||_2^2
        bend_coef tr(w_ng' K_nn w_ng)
        tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
        
        Implementation covers general case where there is a wt_n and bend_coef per dimension
        """
        # expand these
        n, a = self.x_na.shape
        bend_coefs = np.ones(a) * self.bend_coef if np.isscalar(self.bend_coef) else self.bend_coef
        rot_coefs = np.ones(a) * self.rot_coef if np.isscalar(self.rot_coef) else self.rot_coef
        wt_n = self.wt_n
        if wt_n.ndim == 1:
            wt_n = wt_n[:,None]
        if wt_n.shape[1] == 1:
            wt_n = np.tile(wt_n, (1,a))
        
        K_nn = tps_kernel_matrix(self.x_na)
        cost = np.zeros(3)
        
        # matching cost
        cost[0] = np.sum(np.square(self.transform_points(self.x_na) - self.y_ng) * wt_n)
        
        # bending cost
        cost[1] = np.trace(np.diag(bend_coefs).dot(self.w_ng.T.dot(K_nn.dot(self.w_ng))))
        
        # rotation cost
        cost[2] = np.trace((self.lin_ag - np.eye(a)).T.dot(np.diag(rot_coefs).dot((self.lin_ag - np.eye(a)))))
        return cost

def prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm, fwd=True):
    """
    Takes into account outlier source points and normalization of points
    """
    if (fwd):
        wt_n = corr_nm.sum(axis=1)
        if np.any(wt_n == 0):
            inlier = wt_n != 0
            x_nd = x_nd[inlier,:]
            wt_n = wt_n[inlier]
            xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
        else:
            xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        wt_n /= len(x_nd) # normalize by number of points
        return x_nd, xtarg_nd, wt_n
    else:
        wt_m = corr_nm.sum(axis=0)
        if np.any(wt_m == 0):
            inlier = wt_m != 0
            y_md = y_md[inlier,:]
            wt_m = wt_m[inlier]
            ytarg_md = (corr_nm[inlier,:]/wt_m[None,:]).T.dot(x_nd)
        else:
            ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        wt_m /= len(y_md) # normalize by number of points            
        return y_md, ytarg_md, wt_m

def tps_rpm(x_nd, y_md, f_solver_factory=None, 
            n_iter=tpsc.N_ITER, em_iter=tpsc.EM_ITER, 
            reg_init=tpsc.REG[0], reg_final=tpsc.REG[1], 
            rad_init=tpsc.RAD[0], rad_final=tpsc.RAD[1], 
            rot_reg=tpsc.ROT_REG, 
            outlierprior=tpsc.OUTLIER_PRIOR, outlierfrac=tpsc.OURLIER_FRAC, 
            prior_prob_nm=None, plotting=False, plot_cb=None):
    _, d = x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    
    f = ThinPlateSpline(d)
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_nd,axis=0) - np.min(x_nd,axis=0))
    f.lin_ag = np.diag(scale) # align the mins and max
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) * scale  # align the medians
    
    # set up outlier priors for source and target scenes
    n, _ = x_nd.shape
    m, _ = y_md.shape
    x_priors = np.ones(n)*outlierprior
    y_priors = np.ones(m)*outlierprior
    
    # set up custom solver if solver factory is specified
    if f_solver_factory is None:
        fsolve = None
    else:
        fsolve = f_solver_factory.get_solver(x_nd, rot_reg)
    
    for i, (reg, rad) in enumerate(zip(regs, rads)):
        for _ in range(em_iter):
            xwarped_nd = f.transform_points(x_nd)

            dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
            prob_nm = np.exp( -dist_nm / rad )
            if prior_prob_nm != None:
                prob_nm *= prior_prob_nm
            
            corr_nm, _, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac)
            corr_nm += 1e-9
            
            x_nd_inlier, xtarg_nd, wt_n = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm)
    
            if fsolve is None:
                f = ThinPlateSpline.fit_ThinPlateSpline(x_nd_inlier, xtarg_nd, reg, rot_reg, wt_n)
            else:
                fsolve.solve(wt_n, xtarg_nd, reg, f) #TODO: handle ouliers in source and round by BEND_COEF_DIGITS
        
        if plotting and (i%plotting==0 or i==(n_iter-1)):
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
    return f, corr_nm

def tps_rpm_bij(x_nd, y_md, f_solver_factory=None, g_solver_factory=None, 
                n_iter=tpsc.N_ITER, em_iter=tpsc.EM_ITER, 
                reg_init=tpsc.REG[0], reg_final=tpsc.REG[1], 
                rad_init=tpsc.RAD[0], rad_final=tpsc.RAD[1], 
                rot_reg=tpsc.ROT_REG, 
                outlierprior=tpsc.OUTLIER_PRIOR, outlierfrac=tpsc.OURLIER_FRAC, 
                prior_prob_nm=None, plotting=False, plot_cb=None):
    _, d = x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_nd,axis=0) - np.min(x_nd,axis=0))
    f.lin_ag = np.diag(scale) # align the mins and max
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) * scale  # align the medians
    g = ThinPlateSpline(d)
    g.lin_ag = np.diag(1./scale)
    g.trans_g = -np.diag(1./scale).dot(f.trans_g)

    # set up outlier priors for source and target scenes
    n, _ = x_nd.shape
    m, _ = y_md.shape
    x_priors = np.ones(n)*outlierprior
    y_priors = np.ones(m)*outlierprior
    
    # set up custom solver if solver factory is specified
    if f_solver_factory is None:
        fsolve = None
    else:
        fsolve = f_solver_factory.get_solver(x_nd, rot_reg)
    if g_solver_factory is None:
        gsolve = None
    else:
        gsolve = g_solver_factory.get_solver(y_md, rot_reg)
    
    for i, (reg, rad) in enumerate(zip(regs, rads)):
        for _ in range(em_iter):
            xwarped_nd = f.transform_points(x_nd)
            ywarped_md = g.transform_points(y_md)
            
            fwddist_nm = ssd.cdist(xwarped_nd, y_md, 'euclidean')
            invdist_nm = ssd.cdist(x_nd, ywarped_md, 'euclidean')
            
            prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*rad) )
            if prior_prob_nm != None:
                prob_nm *= prior_prob_nm
            
            corr_nm, _, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
            corr_nm += 1e-9
            
            x_nd_inlier, xtarg_nd, wt_n = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm)
            y_md_inlier, ytarg_md, wt_m = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm, fwd=False)
    
            if fsolve is None:
                f = ThinPlateSpline.fit_ThinPlateSpline(x_nd_inlier, xtarg_nd, reg, rot_reg, wt_n)
            else:
                fsolve.solve(wt_n, xtarg_nd, reg, f) #TODO: handle ouliers in source and round by BEND_COEF_DIGITS
            if gsolve is None:
                g = ThinPlateSpline.fit_ThinPlateSpline(y_md_inlier, ytarg_md, reg, rot_reg, wt_m)
            else:
                gsolve.solve(wt_m, ytarg_md, reg, g) #TODO: handle ouliers in source and round by BEND_COEF_DIGITS
        
        if plotting and (i%plotting==0 or i==(n_iter-1)):
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
    
    return f, g, corr_nm

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    if n == 1:
        return np.r_[b]
    else:
        return np.exp(np.linspace(np.log(a),np.log(b),n))

def balance_matrix3(prob_nm, max_iter, row_priors, col_priors, outlierfrac, r_N = None):
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = row_priors
    prob_NM[n, :m] = col_priors
    prob_NM[n, m] = np.sqrt(np.sum(row_priors)*np.sum(col_priors)) # this can `be weighted bigger weight = fewer outliers
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
    
    return prob_NM[:n, :m].astype(np.float64), r_N, c_M
