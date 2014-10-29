from __future__ import division

import scipy.linalg
from tpsopt import precompute

class TpsSolver(object):
    """
    Fits thin plate spline to data using precomputed matrix products
    """
    def __init__(self, bend_coefs, N, QN, NON, NR, x_nd, K_nn, rot_coef):
        for b in bend_coefs:
            if b not in NON:
                raise RuntimeError("No precomputed NON for bending coefficient {}".format(b))
        self.rot_coef = rot_coef
        self.n, self.d  = x_nd.shape
        self.bend_coefs = bend_coefs
        self.N          = N
        self.QN         = QN        
        self.NON        = NON
        self.NR         = NR
        self.x_nd       = x_nd
        self.K_nn       = K_nn
        self.valid = True
    
    def solve(self, wt_n, y_nd, bend_coef, f_res):
        if y_nd.shape[0] != self.n or y_nd.shape[1] != self.d:
            raise RuntimeError("The dimensions of y_nd doesn't match the dimensions of x_nd")
        if bend_coef not in self.bend_coefs:
            raise RuntimeError("No precomputed NON for bending coefficient {}".format(bend_coef))
        WQN = wt_n[:, None] * self.QN
        lhs = self.NON[bend_coef] + self.QN.T.dot(WQN)
        wy_nd = wt_n[:, None] * y_nd
        rhs = self.NR + self.QN.T.dot(wy_nd)
        z = scipy.linalg.solve(lhs, rhs)
        theta = self.N.dot(z)
        f_res.set_ThinPlateSpline(self.x_nd, y_nd, bend_coef, self.rot_coef, wt_n, theta=theta)

class TpsSolverFactory(object):
    def get_solver(self, x_na, K_nn, bend_coefs, rot_coef):
        N, QN, NON, NR = precompute.get_exact_solver(x_na, K_nn, bend_coefs, rot_coef)
        return TpsSolver(bend_coefs, N, QN, NON, NR, x_na, K_nn, rot_coef)
