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
        self._z_ng = value
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

def l2_distance(x_nd, y_md, rad):
    """
    Compute the L2 distance between the two Gaussian mixture densities constructed from a moving 'model' point set and a fixed 'scene' point set at a given 'scale'. The term that only involves the fixed 'scene' is excluded from the returned distance.  The gradient with respect to the 'model' is calculated and returned as well.
    """
    f1, g1 = gauss_transform(x_nd, x_nd, rad)
    f2, g2 = gauss_transform(x_nd, y_md, rad)
    f =  f1 - 2*f2
    g = 2*g1 - 2*g2
    return f, g

def tps_l2_obj(z, QN, NKN, NRN, NR, y_md, rad, reg, rot_reg):
    n = QN.shape[1]
    d = y_md.shape[1]
    z = z.reshape((n, d))
    xwarped_nd = QN.dot(z)
    distance, distance_grad = l2_distance(xwarped_nd, y_md, rad)
    
#     bending = np.trace(z.T.dot(NKN.dot(z)))
#     rotation = np.trace(z.T.dot(NRN.dot(z))) - 2 * np.trace(z.T.dot(NR))
#     energy = distance + reg * bending + rotation
#     grad = QN.T.dot(distance_grad)
#     grad += 2 * reg * NKN.dot(z)
#     grad += 2 * NRN.dot(z) - 2 * NR
#     grad = grad.reshape(d*n)
    
    regNKN_NRN_z = (reg * NKN + NRN).dot(z)
    energy = distance + np.trace(z.T.dot(regNKN_NRN_z)) - 2 * np.trace(z.T.dot(NR))
    grad = QN.T.dot(distance_grad)
    grad += 2 * (reg * NKN + NRN).dot(z) - 2 * NR
    grad = grad.reshape(d*n)

    return energy, grad

def tps_l2(x_ld, y_md, ctrl_nd=None, 
            n_iter=settings.L2_N_ITER, opt_iter=settings.L2_OPT_ITER, 
            reg_init=settings.L2_REG[0], reg_final=settings.L2_REG[1], 
            rad_init=settings.L2_RAD[0], rad_final=settings.L2_RAD[1], 
            rot_reg=settings.L2_ROT_REG, 
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
    NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
    NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
    for reg, rad in zip(regs, rads):
        res = so.fmin_l_bfgs_b(tps_l2_obj, z_nd, None, args=(f.QN_ln, f.NKN_nn, NRN_nn, NR_nd, y_md, rad, reg, rot_reg), maxfun=opt_iter)
        z_nd = res[0]
        f.z_ng = z_nd.reshape((n, d))
        if callback is not None:
            callback(x_ld, y_md, f)
    return f

def multi_tps_l2_obj(z_knd, solver_k, y_kmd, p_ktd, rad, reg, rot_reg, cov_coef, separate_cov=False):
    n_k = np.asarray([solver[0].shape[1] for solver in solver_k])
    k = n_k.shape[0]
    n_start_k = np.r_[0, np.cumsum(n_k)[:-1]]
    n_end_k = np.cumsum(n_k)
    d = y_kmd[0].shape[1]

    total_energy = 0
    total_grad = np.zeros(n_end_k[-1]*d)
    
    for (n_start, n_end, (N, QN, NKN, NRN, NR, QN_tn), y_md) in zip(n_start_k, n_end_k, solver_k, y_kmd):
        assert d == y_md.shape[1]
        z_nd = z_knd[n_start*d:n_end*d]
        energy, grad = tps_l2_obj(z_nd, QN, NKN, NRN, NR, y_md, rad, reg, rot_reg)
        total_energy += energy
        total_grad[n_start*d:n_end*d] += grad

    energy_cov = 0
    grad_cov = 0
    for i in range(p_ktd.shape[1]):
        QN_1kn = []
        for n_start, n_end, (N, QN, NKN, NRN, NR, QN_tn) in zip(n_start_k, n_end_k, solver_k):
            QN_1kn.append(QN_tn[i,:])
        QN_1kn = np.concatenate(QN_1kn)
        for n_start, n_end, (N, QN, NKN, NRN, NR, QN_tn) in zip(n_start_k, n_end_k, solver_k):
            L_1kn = (-1/k) * QN_1kn
            L_1kn[n_start:n_end] += QN_tn[i,:]
            Lz_1d = L_1kn[None,:].dot(z_knd.reshape((-1,d)))
            energy_cov += (1/k) * np.sum(np.square(Lz_1d))
            grad_cov += (1/k) * 2 * L_1kn[:,None].dot(Lz_1d).reshape(-1)

#     # slow computation of energy_cov
#     fp_ktd = []
#     for (n, n_start, n_end, (N, QN, NKN, NRN, NR, QN_tn), y_md, p_td) in zip(n_k, n_start_k, n_end_k, solver_k, y_kmd, p_ktd):
#         assert d == y_md.shape[1]
#         z_nd = z_knd[n_start*d:n_end*d]
#         z_nd = z_nd.reshape((n, d))
#         fp_td = QN_tn.dot(z_nd)
#         fp_ktd.append(fp_td)
#     fp_ktd = np.array(fp_ktd)
#     k = p_ktd.shape[0]
#     energy_cov2 = 0
#     for i in range(p_ktd.shape[1]):
#         fp_kd = fp_ktd[:,i,:]
#         energy_cov2 += (1/k) * np.trace((fp_kd - fp_kd.mean(axis=0)).T.dot(fp_kd - fp_kd.mean(axis=0)))
#     print "energy_cov equal?", np.allclose(energy_cov, energy_cov2)

    total_energy += cov_coef * energy_cov
    total_grad += cov_coef * grad_cov
    
    return total_energy, total_grad

def multi_tps_l2(x_kld, y_kmd, p_ktd, ctrl_knd=None, 
                   n_iter=settings.L2_N_ITER, opt_iter=settings.L2_OPT_ITER, 
                   reg_init=settings.L2_REG[0], reg_final=settings.L2_REG[1], 
                   rad_init=settings.L2_RAD[0], rad_final=settings.L2_RAD[1], 
                   rot_reg=settings.L2_ROT_REG, 
                   cov_coef=settings.COV_COEF, 
                   callback=None, 
                   multi_callback=None):
    if ctrl_knd is None:
        ctrl_knd = x_kld
    
    # intitalize z from independent optimizations
    f_k = []
    z_knd = []
    solver_k = []
    for (x_ld, y_md, p_td, ctrl_nd) in zip(x_kld, y_kmd, p_ktd, ctrl_knd):
        n, d = ctrl_nd.shape
        f = tps_l2(x_ld, y_md, ctrl_nd=ctrl_nd, n_iter=n_iter, opt_iter=opt_iter, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final, rot_reg=rot_reg, callback=callback)
        f_k.append(f)
        NR_nd = f.N_bn[1:1+d, :].T * rot_reg[:d]
        NRN_nn = NR_nd.dot(f.N_bn[1:1+d, :])
        t = p_td.shape[0]
        K_tn = tps_kernel_matrix2(p_td, ctrl_nd)
        Q_tb = np.c_[np.ones((t, 1)), p_td, K_tn]
        QN_tn = Q_tb.dot(f.N_bn)
        solver = (f.N_bn, f.QN_ln, f.NKN_nn, NRN_nn, NR_nd, QN_tn)
        z_knd.append(f.z_ng.reshape(n*d))
        solver_k.append(solver)
    z_knd = np.concatenate(z_knd)
    n_k = np.asarray([solver[0].shape[1] for solver in solver_k])
    n_start_k = np.r_[0, np.cumsum(n_k)[:-1]]
    n_end_k = np.cumsum(n_k)

    for (n, n_start, n_end, f) in zip(n_k, n_start_k, n_end_k, f_k):
        f.z_ng = z_knd[n_start*d:n_end*d].reshape((n, d))
    if multi_callback is not None:
        multi_callback(x_kld, y_kmd, f_k)
    
    res = so.fmin_l_bfgs_b(multi_tps_l2_obj, z_knd, None, args=(solver_k, y_kmd, p_ktd, rad_final, reg_final, rot_reg, cov_coef), maxfun=opt_iter)
    z_knd = res[0]
    
    for (n, n_start, n_end, f) in zip(n_k, n_start_k, n_end_k, f_k):
        f.z_ng = z_knd[n_start*d:n_end*d].reshape((n, d))
    if multi_callback is not None:
        multi_callback(x_kld, y_kmd, f_k)
    
#     # check gradients
#     energy, grad = l2_tps_multi_obj(z_knd, solver_k, y_kmd, p_ktd, rads[-1], regs[-1], rot_reg, cov_coef)
#     energy_fun = lambda z_knd: l2_tps_multi_obj(z_knd, solver_k, y_kmd, p_ktd, rads[-1], regs[-1], rot_reg, cov_coef)[0]
#     import numdifftools
#     energy_dfun = numdifftools.Gradient(energy_fun)
#     num_grad = energy_dfun(z_knd)
#     print "grad equal?", np.allclose(grad, num_grad)
#     import IPython as ipy
#     ipy.embed()

    return f_k
