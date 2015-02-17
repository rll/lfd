from __future__ import division

import settings
import numpy as np
import scipy.optimize as so
from transformation import Transformation
from tps import tps_kernel_matrix, tps_kernel_matrix2, tps_grad, loglinspace

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
        if self._theta_bg is None:
            self._theta_bg = self.N_bn.dot(self.z_ng)
        return self._theta_bg
    
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
            n_iter=settings.N_ITER, opt_iter=100, 
            reg_init=settings.REG[0], reg_final=settings.REG[1], 
            rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
            rot_reg=settings.ROT_REG, 
            callback=None):
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
            callback(f, y_md)
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

def multi_tps_l2_obj(z_knd, f_k, L_ktkn, y_md, p_ktd, rad, reg, rot_reg, cov_coef):
    f_k = params_to_multi_tps(z_knd, f_k)

    pw_tps_l2_energy, pw_tps_l2_grad_knd = pairwise_tps_l2_obj(z_knd, f_k, y_md, rad, reg, rot_reg)

    _, d = y_md.shape
    Lz_ktd = L_ktkn.dot(z_knd.reshape((-1,d)))
    cov_energy = np.sum(np.square(Lz_ktd))
    cov_grad_knd = 2 * L_ktkn.T.dot(Lz_ktd).reshape(-1)

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

def multi_tps_l2(x_kld, y_md, p_ktd, ctrl_knd=None, 
            n_iter=settings.N_ITER, opt_iter=100, 
            reg_init=settings.REG[0], reg_final=settings.REG[1], 
            rad_init=settings.RAD[0], rad_final=settings.RAD[1], 
            rot_reg=settings.ROT_REG, 
            cov_coef=settings.COV_COEF, 
            callback=None, 
            multi_callback=None):
    if ctrl_knd is None:
        ctrl_knd = x_kld
    
    # intitalize z from independent optimizations
    f_k = []
    QN_ktn = []
    for (x_ld, p_td, ctrl_nd) in zip(x_kld, p_ktd, ctrl_knd):
        n, d = ctrl_nd.shape
        f = tps_l2(x_ld, y_md, ctrl_nd=ctrl_nd, n_iter=n_iter, opt_iter=opt_iter, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final, rot_reg=rot_reg, callback=callback)
        f_k.append(f)
        QN_tn = f.compute_transform_grad(p_td)
        QN_ktn.append(QN_tn)
    z_knd = multi_tps_to_params(f_k)

    # put together matrix for computing sum of variances
    # the sum of variances is given by np.sum(np.square(L_ktkn.dot(z_knd.reshape((-1,d)))))
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
    L_ktkn = (1/k) * np.array(L_ktkn)

    if multi_callback is not None:
        multi_callback(f_k, y_md, p_ktd)

    res = so.fmin_l_bfgs_b(multi_tps_l2_obj, z_knd, None, args=(f_k, L_ktkn, y_md, p_ktd, rad_final, reg_final, rot_reg, cov_coef), maxfun=opt_iter)
    z_knd = res[0]

    f_k = params_to_multi_tps(z_knd, f_k)    
    if multi_callback is not None:
        multi_callback(f_k, y_md, p_ktd)

    return f_k
