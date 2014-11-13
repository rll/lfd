from __future__ import division

import numpy as np

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import scikits.cuda.linalg as culinalg
    culinalg.init()
    _has_cuda = True
except (ImportError, OSError):
    _has_cuda = False

def balance_matrix3_gpu(prob_nm, max_iter, row_priors, col_priors, outlierfrac, r_N = None):
    if not _has_cuda:
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
