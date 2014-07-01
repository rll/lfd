"""
Slightly faster way to compute lots of svds
"""


import numpy as np


def svds(x_k33):
    K = len(x_k33)
    option = "A"
    m = 3
    n = 3
    nvt = 3
    lwork = 575
    iwork = np.zeros(24, 'int32')
    work = np.zeros((575))
    s_k3 = np.zeros((K,3))
    fu_k33 = np.zeros((K,3,3))
    fvt_k33 = np.zeros((K,3,3))
    fx_k33 = x_k33.transpose(0,2,1).copy()
    lapack_routine = np.linalg.lapack_lite.dgesdd
    for k in xrange(K):
        lapack_routine(option, m, n, fx_k33[k], m, s_k3[k], fu_k33[k], m, fvt_k33[k], nvt,
                       work, lwork, iwork, 0)    
    return fu_k33.transpose(0,2,1), s_k3, fvt_k33.transpose(0,2,1)


def svds_slow(x_k33):
    s2,u2,v2 = [],[],[]
    for x_33 in x_k33:
        u,s,vt = np.linalg.svd(x_33)
        s2.append(s)
        u2.append(u)
        v2.append(vt)
    s2 = np.array(s2)
    u2 = np.array(u2)
    v2 = np.array(v2)
    return u2,s2,v2

def test_svds():
    x_k33 = np.random.randn(1000,3,3)
    
    u1,s1,v1 = svds(x_k33)
    u2,s2,v2 = svds_slow(x_k33)
    assert np.allclose(u1,u2)
    assert np.allclose(s1,s2)
    assert np.allclose(v1,v2)
if __name__ == "__main__":
    test_svds()