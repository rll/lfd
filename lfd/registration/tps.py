"""
Functions for fitting and applying thin plate spline transformations
"""
from __future__ import division

import settings
import numpy as np
import scipy.spatial.distance as ssd
from transformation import Transformation
import lfd.registration
if lfd.registration._has_cuda:
    import pycuda.gpuarray as gpuarray
    import scikits.cuda.linalg as culinalg

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

def solve_eqp1(H, f, A, ret_factorization=False):
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
    
    if ret_factorization:
        return x, (N, z)
    return x

def tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n, ret_factorization=False):
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
        
        if ret_factorization:
            theta, (N, z) = solve_eqp1(H, f, A, ret_factorization=True)
        else:
            theta = solve_eqp1(H, f, A)
    else:
        bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
        if wt_n.ndim == 1:
            wt_n = wt_n[:,None]
        if wt_n.shape[1] == 1:
            wt_n = np.tile(wt_n, (1,d))
        theta = np.empty((1+d+n,d))
        z = np.empty((n,d))
        for i in range(d):
            WQ = wt_n[:,i][:,None] * Q
            QWQ = Q.T.dot(WQ)
            H = QWQ
            H[d+1:,d+1:] += bend_coefs[i] * K_nn
            H[1:d+1, 1:d+1] += np.diag(rot_coefs)
             
            f = -WQ.T.dot(y_ng[:,i])
            f[1+i] -= rot_coefs[i]
            
            if ret_factorization:
                theta[:,i], (N, z[:,i]) = solve_eqp1(H, f, A, ret_factorization=True)
            else:
                theta[:,i] = solve_eqp1(H, f, A)
    
    if ret_factorization:
        return theta, (N, z)
    return theta

class ThinPlateSpline(Transformation):
    """
    Attributes:
        x_na: centers of basis functions
        w_ng: weights of basis functions
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    """
    def __init__(self, d=3):
        "initialize as identity"
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))
        self.N = None
        self.z = None
        
        self.y_ng = np.zeros((0,d))
        self.bend_coef = 0
        self.rot_coef = 0
        self.wt_n = np.zeros(0)
    
    @staticmethod
    def create_from_optimization(x_na, y_ng, bend_coef, rot_coef, wt_n):
        r"""Solves the optimization problem
            
        .. math::
            :nowrap:
    
            \begin{align*}
                & \min_{f} 
                    & \sum_{i=1}^n w_i ||y_i - f(x_i)||_2^2
                    + \lambda Tr(A^\top K A)
                    + Tr((B - I) R (B - I)) \\
                & \text{subject to} 
                    &  X^\top A = 0 \\
                    && 1^\top A = 0 \\
            \end{align*}
        
        Args:
            x_na: source cloud, :math:`X`
            y_ng: target cloud, :math:`Y`
            bend_coef: smoothing, penalize non-affine part, :math:`\lambda`
            rot_coef: angular_spring, penalize rotation, :math:`\text{diag}(R)`
            wt_n: weight the points, :math:`w`
        
        Returns:
            A ThinPlateSpline f
        """
        f = ThinPlateSpline()
        theta, (N, z) = tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n, ret_factorization=True)
        f.update(x_na, y_ng, bend_coef, rot_coef, wt_n, theta, N=N, z=z)
        return f

    def update(self, x_na, y_ng, bend_coef, rot_coef, wt_n, theta, N=None, z=None):
        d = x_na.shape[1]
        self.trans_g = theta[0]
        self.lin_ag = theta[1:d+1]
        self.w_ng = theta[d+1:]
        self.N = N
        self.z = z
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
        r"""Returns the following 3 objectives:
        
            - :math:`\sum_{i=1}^n w_i ||y_i - f(x_i)||_2^2`
            - :math:`\lambda Tr(A^\top K A)`
            - :math:`Tr((B - I) R (B - I))`
        
        Note:
            Implementation covers general case where there is a wt_n and bend_coef per dimension
        """
        # expand these
        _, a = self.x_na.shape
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
            xtarg_nd = np.zeros_like(x_nd)
            xtarg_nd[inlier,:] = (corr_nm[inlier,:]/wt_n[inlier,None]).dot(y_md)
        else:
            xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        wt_n /= len(x_nd) # normalize by number of points
        return xtarg_nd, wt_n
    else:
        wt_m = corr_nm.sum(axis=0)
        if np.any(wt_m == 0):
            inlier = wt_m != 0
            ytarg_md = np.zeros_like(y_md)
            ytarg_md[inlier,:] = (corr_nm[inlier,:]/wt_m[None,inlier]).T.dot(x_nd)
        else:
            ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        wt_m /= len(y_md) # normalize by number of points
        return ytarg_md, wt_m

def tps_rpm(x_nd, y_md, f_solver_factory=None, 
            n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
            reg_init=settings.REG[0], reg_final=settings.REG[1], 
            rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
            rot_reg=settings.ROT_REG, 
            outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OURLIER_FRAC, 
            prior_prob_nm=None, callback=None):
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
        for i_em in range(em_iter):
            xwarped_nd = f.transform_points(x_nd)

            dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
            prob_nm = np.exp( -dist_nm / (2*rad) )
            if prior_prob_nm != None:
                prob_nm *= prior_prob_nm
            
            corr_nm, _, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac)
            
            xtarg_nd, wt_n = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm)
            if fsolve is None:
                f = ThinPlateSpline.create_from_optimization(x_nd, xtarg_nd, reg, rot_reg, wt_n)
            else:
                fsolve.solve(wt_n, xtarg_nd, reg, f)
            
            if callback:
                callback(i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad)
        
    return f, corr_nm

def tps_rpm_bij(x_nd, y_md, f_solver_factory=None, g_solver_factory=None, 
                n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
                reg_init=settings.REG[0], reg_final=settings.REG[1], 
                rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
                rot_reg=settings.ROT_REG, 
                outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OURLIER_FRAC, 
                prior_prob_nm=None, callback=None):
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
        for i_em in range(em_iter):
            xwarped_nd = f.transform_points(x_nd)
            ywarped_md = g.transform_points(y_md)
            
            fwddist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
            invdist_nm = ssd.cdist(x_nd, ywarped_md, 'sqeuclidean')
            
            prob_nm = np.exp( -((1/n) * fwddist_nm + (1/m) * invdist_nm) / (2*rad * (1/n + 1/m)) )
            if prior_prob_nm != None:
                prob_nm *= prior_prob_nm
            
            corr_nm, _, _ =  balance_matrix3(prob_nm, 10, x_priors, y_priors, outlierfrac) # edit final value to change outlier percentage
            
            xtarg_nd, wt_n = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm)
            ytarg_md, wt_m = prepare_fit_ThinPlateSpline(x_nd, y_md, corr_nm, fwd=False)
    
            if fsolve is None:
                f = ThinPlateSpline.create_from_optimization(x_nd, xtarg_nd, reg, rot_reg, wt_n)
            else:
                fsolve.solve(wt_n, xtarg_nd, reg, f)
            if gsolve is None:
                g = ThinPlateSpline.create_from_optimization(y_md, ytarg_md, reg, rot_reg, wt_m)
            else:
                gsolve.solve(wt_m, ytarg_md, reg, g)
            
            if callback:
                callback(i, i_em, x_nd, y_md, xtarg_nd, corr_nm, wt_n, f, g, corr_nm, rad)
    
    return f, g, corr_nm

def loglinspace(start, stop, num):
    """Return numbers spaced with a constant ratio.

    Returns `num` numbers in the interval [`start`, `stop`],
    with constant ratio between consecutive numbers.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence.
        num: Number of samples to generate.

    Note:
        Unlike np.linspace, a singleton sequence with `stop`
        is returned when `num` is 1.

    Example:

        >>> loglinspace(1.0, 100.0, 3)
        array([   1.,   10.,  100.])
        >>> loglinspace(10.0, 0.001, 5)
        array([  1.00000000e+01,   1.00000000e+00,   1.00000000e-01,
                 1.00000000e-02,   1.00000000e-03])
        >>> loglinspace(2, 4, 1)
        array([ 4.])
    """
    if num == 1:
        return np.array([stop]).astype(np.float64)
    else:
        return np.exp(np.linspace(np.log(start), np.log(stop), num))

def balance_matrix3_cpu(prob_nm, max_iter, row_priors, col_priors, outlierfrac, r_N = None):
    """Balances matrix, including the prior row and column.
    
    Example:
    
        >>> from lfd.registration.tps import balance_matrix3_cpu
        >>> import numpy as np
        >>> n, m = (100, 150)
        >>> prob_nm = np.random.random((n,m))
        >>> p_n = 0.1 * np.random.random(n)
        >>> p_m = 0.1 * np.random.random(m)
        >>> outlierfrac = 1e-2
        >>> prob_nm0 = balance_matrix3_cpu(prob_nm, 10, p_n, p_m, outlierfrac)[0]
        >>> prob_NM = np.empty((n+1, m+1))
        >>> prob_NM[:n, :m] = prob_nm.copy()
        >>> prob_NM[:n, m] = p_n.copy()
        >>> prob_NM[n, :m] = p_m.copy()
        >>> prob_NM[n, m] = np.sqrt(np.sum(p_n)*np.sum(p_m))
        >>> a_N = np.r_[np.ones(n), m*outlierfrac]
        >>> b_M = np.r_[np.ones(m), n*outlierfrac]
        >>> for _ in xrange(10):
        ...     prob_NM = prob_NM / (prob_NM.sum(axis=0) / b_M)[None, :]
        ...     prob_NM = prob_NM / (prob_NM.sum(axis=1) / a_N)[:, None]
        ... 
        >>> prob_nm1 = prob_NM[:n,:m]
        >>> np.allclose(prob_nm0, prob_nm1)
        True
    """
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

def balance_matrix3_gpu(prob_nm, max_iter, row_priors, col_priors, outlierfrac, r_N = None):
    if not lfd.registration._has_cuda:
        raise NotImplementedError("CUDA not installed")
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
    
    if r_N is None: r_N = np.ones((n+1,1),'f4')
    
    prob_NM_gpu = gpuarray.empty((n+1,m+1), dtype=np.float32)
    prob_MN_gpu = gpuarray.empty((m+1,n+1), dtype=np.float32)
    r_N_gpu = gpuarray.empty((n+1,1), dtype=np.float32)
    c_M_gpu = gpuarray.empty((m+1,1), dtype=np.float32)
    prob_NM_gpu.set_async(prob_NM)
    prob_MN_gpu.set_async(prob_NM.T.copy())
    r_N_gpu.set_async(r_N)
    
    for _ in xrange(max_iter):
        culinalg.dot(prob_NM_gpu, r_N_gpu, transa='T', out=c_M_gpu)
        c_M_gpu.set_async(b_M[:,None]/c_M_gpu.get())
        culinalg.dot(prob_MN_gpu, c_M_gpu, transa='T', out=r_N_gpu)
        r_N_gpu.set_async(a_N[:,None]/r_N_gpu.get())

    r_N = r_N_gpu.get()
    c_M = c_M_gpu.get()
    prob_NM *= r_N
    prob_NM *= c_M.T
    
    return prob_NM[:n, :m].astype(np.float64), r_N, c_M

def balance_matrix4(prob_nm, max_iter, p_n, p_m):
    """Like balance_matrix3 but doesn't normalize the p_m row and the p_n column
    
    Example:

        >>> from lfd.registration.tps import balance_matrix4
        >>> import numpy as np
        >>> n, m = (100, 150)
        >>> prob_nm = np.random.random((n,m))
        >>> p_n = 0.1 * np.random.random(n)
        >>> p_m = 0.1 * np.random.random(m)
        >>> prob_nm0 = balance_matrix4(prob_nm, 10, p_n, p_m)
        >>> prob_nm1 = prob_nm.copy()
        >>> for _ in xrange(10):
        ...     prob_nm1 = prob_nm1 / (prob_nm1.sum(axis=0) + p_m)[None, :]
        ...     prob_nm1 = prob_nm1 / (prob_nm1.sum(axis=1) + p_n)[:, None]
        ... 
        >>> np.allclose(prob_nm0, prob_nm1)
        True
    """
    n,m = prob_nm.shape
    p_n = p_n.astype('f4')
    p_m = p_m.astype('f4')
    a_n = np.ones(n,'f4')
    b_m = np.ones(m,'f4')
    
    r_n = np.ones(n,'f4')
    c_m = np.ones(m,'f4')
    prob_nm = prob_nm.astype('f4')
    for _ in xrange(max_iter):
        c_m = b_m/(r_n.dot(prob_nm) + p_m/c_m)
        r_n = a_n/(prob_nm.dot(c_m) + p_n/r_n)
    prob_nm *= r_n[:,None]
    prob_nm *= c_m[None,:]
    
    return prob_nm.astype(np.float64)

def balance_matrix3(*args, **kwargs):
    if lfd.registration._has_cuda:
        ret = balance_matrix3_gpu(*args, **kwargs)
    else:
        ret = balance_matrix3_cpu(*args, **kwargs)
    return ret
