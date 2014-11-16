import h5py
import numpy as np
import scipy.linalg
import argparse

from lfd.rapprentice.tps import tps_kernel_matrix 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str)
    parser.add_argument('--bend_coeff_init', type=float, default=10)
    parser.add_argument('--bend_coeff_final', type=float, default=.1)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

def get_lu_decomp(x_na, bend_coef, rot_coef):
    """
    precomputes the LU decomposition and other intermediate results needed
    to fit a TPS to x_na with bend_coeff

    all thats needed is to compute the righthand side and do a forward solve
    """
    
    n,d = x_na.shape
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    QWQ = Q.T.dot(WQ)
    H = Q.T.dot(Q)
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T

    # f = -WQ.T.dot(y_ng)
    # f[1:d+1,0:d] -= np.diag(rot_coefs)

    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]

    p, l, u, = scipy.linalg.lu(N.T.dot(H.dot(N)))

    # z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    # x = N.dot(z)

    res_dict = {'p' : p, 'l' : l, 'u' : u, 'N' : N, 'rot_coeffs' : rot_coeffs}

    return bend_coeff, res_dict


def main():
    args = parse_arguments()

    f = h5py.File(args.datafile, 'r')
    
    bend_coefs = np.loglinspace(args.bend_coef_init, args.bend_coeff_final, args.n_iter)

    for seg_name, seg_info in f.iteritems():
        if 'LU' not in seg_info:
            lu_group = seg_info.create_group('LU')
        else:
            lu_group = seg_info['LU']
        x_na = seg_info['cloud_xyz'][:]
        for bend_coeff in bend_coeffs:
            if str(bend_coeff) in lu_group:
                continue
            
            bend_coeff_g = lu_group.create_group(str(bend_coeff))
            _, res = get_lu_decomp(x_na, bend_coeff)
            for k, v in res.iteritems():
                bend_coeff_g[k] = v

            if args.verbose:
                print 'segment {}  bend_coeff {}'.format(seg_name, bend_coeff)

    f.close()

if __name__=='__main__':
    main()
            
