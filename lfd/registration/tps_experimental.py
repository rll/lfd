from __future__ import division

import settings
import numpy as np
import scipy.spatial.distance as ssd
import scipy.optimize as so
import transformation
from transformation import Transformation
from tps import tps_kernel_matrix, tps_kernel_matrix2, tps_grad, loglinspace, nan2zero, prepare_fit_ThinPlateSpline, balance_matrix3

class ThinPlateSpline(Transformation):
    """
    Attributes:
        x_na: centers of basis functions
        w_ng: weights of basis functions
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    """
    def __init__(self, x_la, ctrl_na, g=None):
        """Inits ThinPlateSpline with identity transformation
        
        Args:
            x_la: source points
            ctrl_na: control points (i.e. ceter of basis functions)
            g: dimension of a target point. Default is the same as the dimension of a source point.
        """
        l, a = x_la.shape
        n = ctrl_na.shape[0]
        assert a == ctrl_na.shape[1]
        if g is None:
            g = a
        
        self.x_la = x_la
        K_ln = tps_kernel_matrix2(x_la, ctrl_na)
        self.Q_lb = np.c_[np.ones((l, 1)), x_la, K_ln]
        self.N_bn = self.compute_N(ctrl_na)
        self.QN_ln = self.Q_lb.dot(self.N_bn)
        
        self.ctrl_na = ctrl_na
        self.K_nn = tps_kernel_matrix(ctrl_na)
        self.NKN_nn = self.N_bn[a+1:, :].T.dot(self.K_nn.dot(self.N_bn[a+1:, :]))
        
        trans_g = np.zeros(g)
        lin_ag = np.eye(a, g)
        self._z_ng = None
        self.z_ng = np.r_[trans_g[None, :], lin_ag, np.zeros((n-a-1, g))]
    
    @property
    def trans_g(self):
        return self.z_ng[0]
    
    @trans_g.setter
    def trans_g(self, value):
        self.z_ng[0] = value
    
    @property
    def lin_ag(self):
        a = self.ctrl_na.shape[1]
        return self.z_ng[1:1+a]
    
    @lin_ag.setter
    def lin_ag(self, value):
        a = self.ctrl_na.shape[1]
        self.z_ng[1:1+a] = value
    
    @property
    def w_ng(self):
        """Can get w_ng but not set it due to change of variables"""
        a = self.ctrl_na.shape[1]
        return self.theta_bg[1+a:]

    @property
    def z_ng(self):
        return self._z_ng
    
    @z_ng.setter
    def z_ng(self, value):
        if self._z_ng is None or self._z_ng.shape == value.shape:
            self._z_ng = value
        else:
            try:
                self._z_ng = value.reshape(self._z_ng.shape) # should raise exception if size changes
            except ValueError:
                raise ValueError("total size of z_ng must be unchanged")
        self._theta_bg = None # indicates it is dirty
    
    @property
    def theta_bg(self):
        """Can get theta_bg but not set it due to change of variables"""
        # TODO: this is incorrect when z_ng is changed in-place
        # if self._theta_bg is None:
        #     self._theta_bg = self.N_bn.dot(self.z_ng)
        # return self._theta_bg
        # return self.N_bn.dot(self.z_ng)
        return self.N_bn.dot(self.z_ng)
    
    @staticmethod
    def compute_N(ctrl_na):
        r"""Computes change of variable matrix
        
        The matrix :math:`N` changes from :math:`z` to :math:`\theta`,
        
        .. math:: \theta = N z
       
        such that the affine part of :math:`\theta` remains unchanged and the 
        non-affine part :math:`A` satisfies the TPS constraint,
        
        
        .. math::
            C^\top A &= 0 \\
            1^\top A &= 0
        
        Args:
            ctrl_na: control points, :math:`C`
        
        Returns:
            N_bn: change of variable matrix, :math:`N`
        
        Example:

            >>> import numpy as np
            >>> from lfd.registration.tps_experimental import ThinPlateSpline
            >>> n = 100
            >>> a = 3
            >>> g = 3
            >>> ctrl_na = np.random.random((n, a))
            >>> z_ng = np.random.random((n, g))
            >>> N_bn = ThinPlateSpline.compute_N(ctrl_na)
            >>> theta_bg = N_bn.dot(z_ng)
            >>> trans_g = theta_bg[0]
            >>> lin_ag = theta_bg[1:a+1]
            >>> w_ng = theta_bg[a+1:]
            >>> print np.allclose(trans_g, z_ng[0])
            True
            >>> print np.allclose(lin_ag, z_ng[1:a+1])
            True
            >>> print np.allclose(ctrl_na.T.dot(w_ng), np.zeros((a, g)))
            True
            >>> print np.allclose(np.ones((n, 1)).T.dot(w_ng), np.zeros((1, g)))
            True
        """
        n, a = ctrl_na.shape
        _u,_s,_vh = np.linalg.svd(np.c_[np.ones((n, 1)), ctrl_na])
        N_bn = np.eye(n+a+1, n)
        N_bn[a+1:, a+1:] = _u[:, a+1:]
        return N_bn
    
    def transform_points(self, x_ma=None):
        """Transforms the x_ma points. If x_ma is not specified, the source points x_la are used."""
        if x_ma is None:
            y_lg = self.QN_ln.dot(self.z_ng)
            return y_lg
        else:
            m = x_ma.shape[0]
            K_mn = tps_kernel_matrix2(x_ma, self.ctrl_na)
            Q_mb = np.c_[np.ones((m, 1)), x_ma, K_mn]
            y_mg = Q_mb.dot(self.theta_bg)
            return y_mg
    
    def compute_transform_grad(self, x_ma=None):
        """Gradient of the transform of the x_ma points. If x_ma is not specified, the source points x_la are used."""
        if x_ma is None:
            return self.QN_ln
        else:
            m = x_ma.shape[0]
            K_mn = tps_kernel_matrix2(x_ma, self.ctrl_na)
            Q_mb = np.c_[np.ones((m, 1)), x_ma, K_mn]
            QN_mn = Q_mb.dot(self.N_bn)
            return QN_mn

    def compute_jacobian(self, x_ma):
        grad_mga = tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.ctrl_na)
        return grad_mga
    
    def get_bending_energy(self):
        return np.trace(self.z_ng.T.dot(self.NKN_nn.dot(self.z_ng)))


def solve_qp(H, f):
    """solve unconstrained qp
    min .5 tr(x'Hx) + tr(f'x)
    """
    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars

    x = np.linalg.solve(H, -f)
    return x

def tpsn_fit(f, y_lg, v_rg, bend_coef, rot_coef, wt_l, wt_r):
    l, r, a = f.l, f.r, f.a
    g = y_lg.shape[1]
    assert y_lg.shape == (l, g)
    assert v_rg.shape == (r, g)
    if wt_l is None: wt_l = np.ones(l)
    if wt_r is None: wt_r = np.ones(r)
    rot_coef = np.ones(a) * rot_coef if np.isscalar(rot_coef) else rot_coef
    assert len(rot_coef) == a
    assert a == g

    WQ0N_le = wt_l[:,None] * f.Q0N_le
    WQ1N_re = wt_r[:,None] * f.Q1N_re
    NR_ea = f.N_be[1:1+a,:].T * rot_coef

    H_ee = f.Q0N_le.T.dot(WQ0N_le) + f.Q1N_re.T.dot(WQ1N_re)
    H_ee += bend_coef * f.NKN_ee
    H_ee += NR_ea.dot(f.N_be[1:1+a,:])

    f_eg = -WQ0N_le.T.dot(y_lg) - WQ1N_re.T.dot(v_rg)
    f_eg -= NR_ea

    f.z_eg = solve_qp(H_ee, f_eg)

def tpsn_rpm(x_ld, u_rd, z_rd, y_md, v_sd, z_sd, 
             n_iter=settings.N_ITER, em_iter=settings.EM_ITER, 
             reg_init=settings.REG[0], reg_final=settings.REG[1], 
             rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
             radn_init=settings.RADN[0], radn_final=settings.RADN[1], 
             nu_init=settings.NU[0], nu_final=settings.NU[1], 
             rot_reg=settings.ROT_REG, 
             outlierprior=settings.OUTLIER_PRIOR, outlierfrac=settings.OUTLIER_FRAC, 
             callback=None, args=()):
    """
    TODO: hyperparameters
    """
    _, d = x_ld.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    radns = loglinspace(radn_init, radn_final, n_iter)
    nus = loglinspace(nu_init, nu_final, n_iter)
    
    f = ThinPlateSplineNormal(x_ld, u_rd, z_rd, x_ld, u_rd, z_rd)
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_ld,axis=0) - np.min(x_ld,axis=0))
    f.lin_ag = np.diag(scale) # align the mins and max
    f.trans_g = np.median(y_md,axis=0) - np.median(x_ld,axis=0) * scale  # align the medians
    
    # set up outlier priors for source and target scenes
    l, _ = x_ld.shape
    m, _ = y_md.shape
    r, _ = u_rd.shape
    s, _ = v_sd.shape
    x_priors = np.ones(l)*outlierprior
    y_priors = np.ones(m)*outlierprior
    u_priors = np.ones(r)*outlierprior
    v_priors = np.ones(s)*outlierprior
    
    for i, (reg, rad, radn, nu) in enumerate(zip(regs, rads, radns, nus)):
        for i_em in range(em_iter):
            xwarped_ld = f.transform_points()
            uwarped_rd = f.transform_vectors()
            zwarped_rd = f.transform_points(z_rd)
            
            beta_r = np.linalg.norm(uwarped_rd, axis=1)

            dist_lm = ssd.cdist(xwarped_ld, y_md, 'sqeuclidean')
            prob_lm = np.exp( -dist_lm / (2*rad) )
            corr_lm, _, _ =  balance_matrix3(prob_lm, 10, x_priors, y_priors, outlierfrac)

            dist_rs = ssd.cdist(uwarped_rd / beta_r[:,None], v_sd, 'sqeuclidean')
            site_dist_rs = ssd.cdist(zwarped_rd, z_sd, 'sqeuclidean')
            prior_prob_rs = np.exp( -site_dist_rs / (2*rad) )
            prob_rs = prior_prob_rs * np.exp( -dist_rs / (2*radn) )
            corr_rs, _, _ =  balance_matrix3(prob_rs, 10, u_priors, v_priors, outlierfrac)

            xtarg_ld, wt_l = prepare_fit_ThinPlateSpline(x_ld, y_md, corr_lm)
            utarg_rd, wt_r = prepare_fit_ThinPlateSpline(u_rd, v_sd, corr_rs)
            wt_r *= nu
            tpsn_fit(f, xtarg_ld, utarg_rd / beta_r[:,None], reg, rot_reg, wt_l, wt_r)

            if callback:
                callback(f, corr_lm, corr_rs, y_md, v_sd, z_sd, xtarg_ld, utarg_rd, wt_l, wt_r, reg, rad, radn, nu, i, i_em, *args)
        
    return f, corr_lm, corr_rs

def tpsn_kernel_matrix2_00(x_la, y_ma):
    distmat_lm = ssd.cdist(x_la, y_ma)
    S00_lm = distmat_lm ** 3
    return S00_lm

def tpsn_kernel_matrix2_01(x_la, u_ra, z_ra):
    distmat_lr = ssd.cdist(x_la, z_ra)
    S01_lr = 3 * distmat_lr
    l, r = S01_lr.shape
    for j in range(r):
        S01_lr[:,j] *= (z_ra[j] - x_la).dot(u_ra[j])
    return S01_lr

def tpsn_kernel_matrix2_11(u_ra, z_ra, v_sa, z_sa):
    distmat_rs = ssd.cdist(z_ra, z_sa)
    S11_rs = -3 / (distmat_rs + 1e-20)
    r, s = S11_rs.shape
    for i in range(r):
        S11_rs[i,:] *= (z_ra[i] - z_sa).dot(u_ra[i])
    for j in range(s):
        S11_rs[:,j] *= (z_ra - z_sa[j]).dot(v_sa[j])
    S11_rs += -3 * distmat_rs * (u_ra.dot(v_sa.T))
    return S11_rs

def tpsn_kernel_matrix2_0(x_la, x_ctrl_na, u_ctrl_ta, z_ctrl_ta):
    S00_ln = tpsn_kernel_matrix2_00(x_la, x_ctrl_na)
    S01_lt = tpsn_kernel_matrix2_01(x_la, u_ctrl_ta, z_ctrl_ta)
    S0_le = np.c_[S00_ln, S01_lt]
    return S0_le

def tpsn_kernel_matrix2_1(u_ra, z_ra, x_ctrl_na, u_ctrl_ta, z_ctrl_ta):
    S10_rn = tpsn_kernel_matrix2_01(x_ctrl_na, u_ra, z_ra).T
    S11_rt = tpsn_kernel_matrix2_11(u_ra, z_ra, u_ctrl_ta, z_ctrl_ta)
    S1_re = np.c_[S10_rn, S11_rt]
    return S1_re

def tpsn_kernel_matrix(x_la, u_ra, z_ra):
    # TODO: specialize this function
    return tpsn_kernel_matrix2(x_la, u_ra, z_ra, x_la, u_ra, z_ra)

def tpsn_kernel_matrix2(x_la, u_ra, z_ra, x_ctrl_na, u_ctrl_ta, z_ctrl_ta):
    S0_le = tpsn_kernel_matrix2_0(x_la, x_ctrl_na, u_ctrl_ta, z_ctrl_ta)
    S1_re = tpsn_kernel_matrix2_1(u_ra, z_ra, x_ctrl_na, u_ctrl_ta, z_ctrl_ta)
    S_ce = np.r_[S0_le, S1_re]
    return S_ce

class ThinPlateSplineNormal(Transformation):
    """
    Attributes:
        x_na: centers of basis functions
        w_ng: weights of basis functions
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    """

    def __init__(self, x_la, u_ra, z_ra, x_ctrl_na, u_ctrl_ta, z_ctrl_ta, g=None):
        """Inits ThinPlateSplineNormal with identity transformation

        Args:
            x_la: source points
            u_ra: source normals
            z_ra: source normal locations
            x_ctrl_na: control points (i.e. center of basis functions)
            u_ctrl_ta: control normals
            z_ctrl_ra: control normal locations
            g: dimension of a target point and normals. Default is the same as the dimension of a source point and normals

        Dimension conventions:
            l: number of source points
            r: number of source normals
            n: number of control points
            t: number of control normals
            a: dimension of source points and normals
            g: dimension of target point and normals
            c: l+r
            e: n+t
            b: e+a+1
        """
        l, a = x_la.shape
        r = u_ra.shape[0]
        assert u_ra.shape[1] == a
        assert z_ra.shape == (r, a)
        n = x_ctrl_na.shape[0]
        assert x_ctrl_na.shape[1] == a
        t = u_ctrl_ta.shape[0]
        assert u_ctrl_ta.shape[1] == a
        assert z_ctrl_ta.shape == (t, a)
        if g is None:
            g = a
        c = l+r
        e = n+t
        b = e+a+1
        self.l = l
        self.r = r
        self.n = n
        self.t = t
        self.a = a
        self.g = g
        self.c = c
        self.e = e
        self.b = b

        self.x_la = x_la
        self.u_ra = u_ra
        self.z_ra = z_ra
        S_ce = tpsn_kernel_matrix2(x_la, u_ra, z_ra, x_ctrl_na, u_ctrl_ta, z_ctrl_ta)
        self.Q0_lb = np.c_[np.ones((l, 1)), x_la, S_ce[:l,:]]
        self.Q1_rb = np.c_[np.zeros((r,1)), u_ra, S_ce[l:,:]]
        self.N_be = self.compute_N(x_ctrl_na, u_ctrl_ta)
        self.Q0N_le = self.Q0_lb.dot(self.N_be)
        self.Q1N_re = self.Q1_rb.dot(self.N_be)

        self.x_ctrl_na = x_ctrl_na
        self.u_ctrl_ta = u_ctrl_ta
        self.z_ctrl_ta = z_ctrl_ta
        self.S_ee = tpsn_kernel_matrix(x_ctrl_na, u_ctrl_ta, z_ctrl_ta)
        D = np.r_[np.c_[np.ones((n, 1)), x_ctrl_na],
                  np.c_[np.zeros((t, 1)), u_ctrl_ta]]
        P_ee = D.dot(np.linalg.inv(D.T.dot(D))).dot(D.T)
        K_ee = np.linalg.inv((np.eye(e) - P_ee).dot(self.S_ee).dot(np.eye(e) - P_ee))
        self.NKN_ee = self.N_be[a+1:, :].T.dot(self.S_ee.dot(self.N_be[a+1:, :]))

        trans_g = np.zeros(g)
        lin_ag = np.eye(a, g)
        self._z_eg = None
        self.z_eg = np.r_[trans_g[None, :], lin_ag, np.zeros((e-a-1, g))]
    
    @property
    def trans_g(self):
        return self.z_eg[0]
    
    @trans_g.setter
    def trans_g(self, value):
        self.z_eg[0] = value
    
    @property
    def lin_ag(self):
        return self.z_eg[1:1+self.a]
    
    @lin_ag.setter
    def lin_ag(self, value):
        self.z_eg[1:1+self.a] = value
    
    @property
    def w_eg(self):
        """Can get w_ng but not set it due to change of variables"""
        return self.theta_bg[1+self.a:]

    @property
    def z_eg(self):
        return self._z_eg
    
    @z_eg.setter
    def z_eg(self, value):
        if self._z_eg is None or self._z_eg.shape == value.shape:
            self._z_eg = value
        else:
            try:
                self._z_eg = value.reshape(self._z_eg.shape) # should raise exception if size changes
            except ValueError:
                raise ValueError("total size of z_eg must be unchanged")
        self._theta_bg = None # indicates it is dirty
    
    @property
    def theta_bg(self):
        """Can get theta_bg but not set it due to change of variables"""
        # TODO: this is incorrect when z_ng is changed in-place
        # if self._theta_bg is None:
        #     self._theta_bg = self.N_be.dot(self.z_eg)
        # return self._theta_bg
        return self.N_be.dot(self.z_eg)
    
    @staticmethod
    def compute_N(x_ctrl_na, u_ctrl_ta):
        r"""Computes change of variable matrix
        
        The matrix :math:`N` changes from :math:`z` to :math:`\theta`,
        
        .. math:: \theta = N z
       
        such that the affine part of :math:`\theta` remains unchanged and the 
        non-affine part :math:`A` satisfies the TPSN constraint,
        
        
        .. math::
            [X^\top U^\top] A &= 0 \\
            [1^\top 0^\top] A &= 0
        
        Args:
            x_ctrl_na: control points, :math:`X`
            u_ctrl_ta: control normals, :math:`U`
        
        Returns:
            N_bn: change of variable matrix, :math:`N`
        """
        n, a = x_ctrl_na.shape
        t, a = u_ctrl_ta.shape
        D = np.r_[np.c_[np.ones((n, 1)), x_ctrl_na],
                  np.c_[np.zeros((t, 1)), u_ctrl_ta]]
        _u,_s,_vh = np.linalg.svd(D)
        N_be = np.eye(n+t+a+1, n+t)
        N_be[a+1:, a+1:] = _u[:, a+1:]
        return N_be
    
    def transform_points(self, x_ma=None):
        """Transforms the x_ma points. If x_ma is not specified, the source points x_la are used."""
        if x_ma is None:
            y_lg = self.Q0N_le.dot(self.z_eg)
            return y_lg
        else:
            m = x_ma.shape[0]
            K0_me = tpsn_kernel_matrix2_0(x_ma, self.x_ctrl_na, self.u_ctrl_ta, self.z_ctrl_ta)
            Q0_mb = np.c_[np.ones((m, 1)), x_ma, K0_me]
            y_mg = Q0_mb.dot(self.theta_bg)
            return y_mg

    def transform_vectors(self, u_sa=None, z_sa=None):
        if (u_sa is None and z_sa is not None) or (u_sa is not None and z_sa is None):
            raise RuntimeError("u_sa and z_sa should both be None or should both be specified")
        if u_sa is None or z_sa is None:
            v_rg = self.Q1N_re.dot(self.z_eg)
            return v_rg
        else:
            s = u_sa.shape[0]
            K1_se = tpsn_kernel_matrix2_1(u_sa, z_sa, self.x_ctrl_na, self.u_ctrl_ta, self.z_ctrl_ta)
            Q1_sb = np.c_[np.zeros((s,1)), u_sa, K1_se]
            v_sg = Q1_sb.dot(self.theta_bg)
            return v_sg

    def compute_transform_grad(self, x_ma=None):
        """Gradient of the transform of the x_ma points. If x_ma is not specified, the source points x_la are used."""
        raise NotImplementedError
        if x_ma is None:
            return self.QN_ln
        else:
            m = x_ma.shape[0]
            S_mn = tps_kernel_matrix2(x_ma, self.ctrl_na)
            Q_mb = np.c_[np.ones((m, 1)), x_ma, S_mn]
            QN_mn = Q_mb.dot(self.N_bn)
            return QN_mn

    def compute_jacobian(self, x_ma):
        # TODO: analytical jacobian is wrong. Use numerical for now
        return np.asarray([self.compute_numerical_jacobian(x_a) for x_a in x_ma])

        n, t, a, g = self.n, self.t, self.a, self.g
        m = x_ma.shape[0]
        assert x_ma.shape[1] == a

        dist_mn = ssd.cdist(x_ma, self.x_ctrl_na, 'euclidean')
        dist_mt = ssd.cdist(x_ma, self.z_ctrl_ta, 'euclidean')
        dot_mt = np.empty((m,t))
        for j in range(t):
            dot_mt[:,j] = (self.z_ctrl_ta[j] - x_ma).dot(self.u_ctrl_ta[j])

        grad_mga = np.empty((m, g, a))

        lin_ga = self.lin_ag.T
        for i in range(a):
            diffi_mn = x_ma[:,i][:,None] - self.x_ctrl_na[:,i][None,:]
            diffi_mt = self.z_ctrl_ta[:,i][None,:] - x_ma[:,i][:,None]
            dS00dx_mn = 3 * (dist_mn ** 2) * diffi_mn
            dS01dx_mt = 3 * (nan2zero(diffi_mt * dot_mt / dist_mt) - (dist_mt * self.u_ctrl_ta[:,i][None,:]))
            grad_mga[:,:,i] = lin_ga[None,:,i] + np.c_[dS00dx_mn, dS01dx_mt].dot(self.w_eg)
        return grad_mga

    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """
        a, g = self.a, self.g
        m, _, d = rot_mad.shape
        newrot_mgd = np.empty((m, g, d))
        for i, (x_a, rot_ad) in enumerate(zip(x_ma, rot_mad)):
            newrot_dg = self.transform_vectors(rot_ad.T, np.tile(x_a, (d, 1)))
            newrot_mgd[i, :, :] = newrot_dg.T

        if orthogonalize:
            if orth_method == "qr":
                newrot_mgd = transformation.orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = transformation.orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = transformation.orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd

    def compute_bending_energy(self, bend_coef=1):
        return bend_coef * np.trace(self.z_eg.T.dot(self.NKN_ee.dot(self.z_eg)))

    def compute_rotation_reg(self, rot_coef=1):
        rot_coef = np.ones(a) * rot_coef if np.isscalar(rot_coef) else rot_coef
        assert len(rot_coef) == self.a
        return np.trace((self.lin_ag - np.eye(self.a)).T.dot(np.diag(rot_coef)).dot(self.lin_ag - np.eye(self.a)))
    
def gauss_transform(A, B, scale):
    m = A.shape[0]
    n = B.shape[0]
    dist = (A[:,None,:] - B[None,:,:])
    sqdist = np.square(dist).sum(2)
    cost = np.exp(-sqdist / (scale**2))
    f = np.sum(cost) / (m * n)
    g = -2. * (cost[:,:,None] * dist).sum(1) / (m * n * (scale**2))
    return f, g

def l2_obj(x_nd, y_md, rad):
    """
    Compute the L2 distance between the two Gaussian mixture densities constructed from a moving 'model' point set and a fixed 'scene' point set at a given 'scale'. The term that only involves the fixed 'scene' is excluded from the returned distance.  The gradient with respect to the 'model' is calculated and returned as well.
    """
    f1, g1 = gauss_transform(x_nd, x_nd, rad)
    f2, g2 = gauss_transform(x_nd, y_md, rad)
    f = f1 - 2*f2
    g = 2*g1 - 2*g2
    return f, g

def tps_l2_obj(z_nd, f, y_md, rad, reg, rot_reg):
    f.z_ng = z_nd
    xwarped_nd = f.transform_points()

    l2_energy, l2_grad_ld = l2_obj(xwarped_nd, y_md, rad)
    energy = l2_energy
    n, d = f.z_ng.shape
    NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
    NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
    energy += np.trace(f.z_ng.T.dot(reg * f.NKN_nn + NRN_nn).dot(f.z_ng)) - 2 * np.trace(f.z_ng.T.dot(NR_nd))
    grad_nd = f.QN_ln.T.dot(l2_grad_ld)
    grad_nd += 2 * (reg * f.NKN_nn + NRN_nn).dot(f.z_ng) - 2 * NR_nd
    grad_nd = grad_nd.reshape(d*n)
    return energy, grad_nd

def tps_l2(x_ld, y_md, ctrl_nd=None, 
            n_iter=settings.L2_N_ITER, opt_iter=settings.L2_OPT_ITER, 
            reg_init=settings.L2_REG[0], reg_final=settings.L2_REG[1], 
            rad_init=settings.L2_RAD[0], rad_final=settings.L2_RAD[1], 
            rot_reg=settings.L2_ROT_REG, 
            callback=None, args=()):
    """TODO: default parameters
    """
    if ctrl_nd is None:
        ctrl_nd = x_ld
    n, d = ctrl_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_ld,axis=0) - np.min(x_ld,axis=0))
    f = ThinPlateSpline(x_ld, ctrl_nd)
    f.lin_ag = np.diag(scale) # align the mins and max1
    f.trans_g = np.median(y_md,axis=0) - np.median(x_ld,axis=0) * scale  # align the medians
    z_nd = f.z_ng.reshape(n*d)

    for reg, rad in zip(regs, rads):
        res = so.fmin_l_bfgs_b(tps_l2_obj, z_nd, None, args=(f, y_md, rad, reg, rot_reg), maxfun=opt_iter)
        z_nd = res[0]
        f.z_ng = z_nd
        if callback is not None:
            callback(f, y_md, *args)
    return f

def pairwise_tps_l2_obj(z_knd, f_k, y_md, rad, reg, rot_reg):
    f_k = params_to_multi_tps(z_knd, f_k)

    energy = 0
    grad_knd = []
    for f in f_k:
        _, d = f.z_ng.shape
        assert d == y_md.shape[1]
        tps_l2_energy, tps_l2_grad_nd = tps_l2_obj(f.z_ng, f, y_md, rad, reg, rot_reg)
        energy += tps_l2_energy
        grad_knd.append(tps_l2_grad_nd)
    grad_knd = np.concatenate(grad_knd)
    return energy, grad_knd

def compute_sum_var_matrix(f_k, p_ktd):
    """Computes the kt by kn matrix L_ktkn for calculating the sum of variances.

    The sum of variances is given by
        (1/k) * np.sum(np.square(L_ktkn.dot(z_knd.reshape((-1,d)))))
    """
    QN_ktn = []
    for f, p_td in zip(f_k, p_ktd):
        QN_tn = f.compute_transform_grad(p_td)
        QN_ktn.append(QN_tn)

    k, t, _ = p_ktd.shape
    L_ktkn = []
    for j in range(t):
        QN_1kn = []
        for QN_tn in QN_ktn:
            QN_1kn.append(QN_tn[j,:])
        QN_1kn = np.concatenate(QN_1kn)
        i = 0
        for QN_tn in QN_ktn:
            _, n = QN_tn.shape
            L_1kn = (-1/k) * QN_1kn
            L_1kn[i:i+n] += QN_tn[j,:]
            L_ktkn.append(L_1kn)
            i += n
    L_ktkn = np.array(L_ktkn)
    return L_ktkn

def tps_cov_obj(z_knd, f_k, p_ktd, L_ktkn=None):
    f_k = params_to_multi_tps(z_knd, f_k)

    if L_ktkn is None:
        L_ktkn = compute_sum_var_matrix(f_k, p_ktd)

    k, t, d = p_ktd.shape
    Lz_ktd = L_ktkn.dot(z_knd.reshape((-1,d)))
    energy = (1/k) * np.sum(np.square(Lz_ktd))
    grad_knd = (1/k) * 2 * L_ktkn.T.dot(Lz_ktd).reshape(-1)

    # fp_ktd = []
    # for f, p_td in zip(f_k, p_ktd):
    #     fp_td = f.transform_points(p_td)
    #     fp_ktd.append(fp_td)
    # fp_ktd = np.array(fp_ktd)
    # energy2 = 0
    # for j in range(t):
    #     fp_kd = fp_ktd[:,j,:]
    #     energy2 += (1/k) * np.trace((fp_kd - fp_kd.mean(axis=0)).T.dot(fp_kd - fp_kd.mean(axis=0)))
    # print "energy cov equal?", np.allclose(energy, energy2)

    return energy, grad_knd

def pairwise_tps_l2_cov_obj(z_knd, f_k, y_md, p_ktd, rad, reg, rot_reg, cov_coef, L_ktkn=None):
    f_k = params_to_multi_tps(z_knd, f_k)

    pw_tps_l2_energy, pw_tps_l2_grad_knd = pairwise_tps_l2_obj(z_knd, f_k, y_md, rad, reg, rot_reg)
    cov_energy, cov_grad_knd = tps_cov_obj(z_knd, f_k, p_ktd, L_ktkn=L_ktkn)
    energy = pw_tps_l2_energy + cov_coef * cov_energy
    grad_knd = pw_tps_l2_grad_knd + cov_coef * cov_grad_knd
    
    return energy, grad_knd

def multi_tps_to_params(f_k):
    z_knd = []
    for f in f_k:
        n, d = f.z_ng.shape
        z_knd.append(f.z_ng.reshape(n*d))
    z_knd = np.concatenate(z_knd)
    return z_knd

def params_to_multi_tps(z_knd, f_k):
    i = 0
    for f in f_k:
        n, d = f.z_ng.shape
        f.z_ng = z_knd[i*d:(i+n)*d]
        i += n
    return f_k

def pairwise_tps_l2_cov(x_kld, y_md, p_ktd, ctrl_knd=None, f_init_k=None, 
                            n_iter=settings.L2_N_ITER, opt_iter=settings.L2_OPT_ITER, 
                            reg_init=settings.L2_REG[0], reg_final=settings.L2_REG[1], 
                            rad_init=settings.L2_RAD[0], rad_final=settings.L2_RAD[1], 
                            rot_reg=settings.L2_ROT_REG, 
                            cov_coef=settings.COV_COEF, 
                            callback=None, args=(), 
                            multi_callback=None, multi_args=()):
    if f_init_k is None:
        if ctrl_knd is None:
            ctrl_knd = x_kld
        else:
            if len(ctrl_knd) != len(x_kld):
                raise ValueError("The number of control points in ctrl_knd is different from the number of point sets in x_kld")
        f_k = []
        # intitalize z from independent optimizations
        f_k = []
        for (x_ld, p_td, ctrl_nd) in zip(x_kld, p_ktd, ctrl_knd):
            n, d = ctrl_nd.shape
            f = tps_l2(x_ld, y_md, ctrl_nd=ctrl_nd, n_iter=n_iter, opt_iter=opt_iter, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final, rot_reg=rot_reg, callback=callback, args=args)
            f_k.append(f)
    else:  
        if len(f_init_k) != len(x_kld):
            raise ValueError("The number of ThinPlateSplines in f_init_k is different from the number of point sets in x_kld")
        f_k = f_init_k
    z_knd = multi_tps_to_params(f_k)

    # put together matrix for computing sum of variances
    L_ktkn = compute_sum_var_matrix(f_k, p_ktd)

    if multi_callback is not None:
        multi_callback(f_k, y_md, p_ktd, *multi_args)

    def opt_multi_callback(z_knd):
        params_to_multi_tps(z_knd, f_k)
        multi_callback(f_k, y_md, p_ktd, *multi_args)

    res = so.fmin_l_bfgs_b(pairwise_tps_l2_cov_obj, z_knd, None, args=(f_k, y_md, p_ktd, rad_final, reg_final, rot_reg, cov_coef, L_ktkn), maxfun=opt_iter, 
                           callback=opt_multi_callback if multi_callback is not None else None)
    z_knd = res[0]

    f_k = params_to_multi_tps(z_knd, f_k)
    if multi_callback is not None:
        multi_callback(f_k, y_md, p_ktd, *multi_args)

    return f_k
