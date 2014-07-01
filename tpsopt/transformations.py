import tps
import numpy as np

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

class Transformation(object):
    """
    Object oriented interface for transformations R^d -> R^d
    """
    def transform_points(self, x_ma):
        raise NotImplementedError
    def compute_jacobian(self, x_ma):
        raise NotImplementedError        

        
    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])
        

        if orthogonalize:
            if orth_method == "qr": 
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
        
    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        hmat_mGD[:,3,:] = np.array([0,0,0,1])
        return hmat_mGD
        
    def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
        "numerical jacobian"
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros(len(x0), len(f0))
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
            dx[i] = 0.
        return jac.transpose()

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

    def transform_points(self, x_ma):
        y_ng = tps.tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return y_ng
    def compute_jacobian(self, x_ma):
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
        
class Affine(Transformation):
    def __init__(self, lin_ag, trans_g):
        self.lin_ag = lin_ag
        self.trans_g = trans_g
    def transform_points(self, x_ma):
        return x_ma.dot(self.lin_ag) + self.trans_g[None,:]  
    def compute_jacobian(self, x_ma):
        return np.repeat(self.lin_ag.T[None,:,:],len(x_ma), axis=0)
        
class Composition(Transformation):
    def __init__(self, fs):
        "applied from first to last (left to right)"
        self.fs = fs
    def transform_points(self, x_ma):
        for f in self.fs: x_ma = f.transform_points(x_ma)
        return x_ma
    def compute_jacobian(self, x_ma):
        grads = []
        for f in self.fs:
            grad_mga = f.compute_jacobian(x_ma)
            grads.append(grad_mga)
            x_ma = f.transform_points(x_ma)
        totalgrad = grads[0]
        for grad in grads[1:]:
            totalgrad = (grad[:,:,:,None] * totalgrad[:,None,:,:]).sum(axis=-2)
        return totalgrad

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    f = ThinPlateSpline()
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.x_na = x_na
    return f

def set_ThinPlateSpline(f, x_na, theta):
    f.x_na = x_na
    d = x_na.shape[1]
    f.trans_g = theta[0]
    f.lin_ag  = theta[1:d+1]
    f.w_ng    = theta[d+1:]
