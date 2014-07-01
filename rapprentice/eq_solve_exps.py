import h5py
import numpy as np
import scipy.sparse, scipy.sparse.linalg
import IPython as ipy
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import scikits.cuda.linalg as culinalg
from rapprentice.culinalg_wrappers import dot

culinalg.init()

@profile
def test_eqs(fname):
    f = h5py.File(fname, 'r')

    #test out with 100 eqs

    lhs = []
    rhs = []

    for i in range(100):
        lhs.append(f[str(i)]['lhs'][:])
        rhs.append(f[str(i)]['rhs'][:])

    full_lhs = scipy.sparse.block_diag(lhs)
    full_rhs = np.vstack(rhs)

    sp_ans = scipy.sparse.linalg.spsolve(full_lhs, full_rhs)

    np_ans = []
    for i in range(100):
        np_ans.append(np.linalg.solve(lhs[i], rhs[i]))
        # l_gpu = gpuarray.to_gpu(lhs[i])
        # r_gpu = gpuarray.to_gpu(rhs[i])
        # culinalg.cho_solve(l_gpu, r_gpu)
    np_ans = np.vstack(np_ans)
    
    assert np.allclose(sp_ans, np_ans)



@profile
def test_hardware():
    N = 15000
    A = np.random.rand(N*N).reshape(N, N)
    A = A.astype('float')
    x = np.random.rand(N)

    A_gpu = gpuarray.to_gpu(A)
    # x_gpu = gpuarray.to_gpu(x)
    b_gpu = dot(A_gpu, A_gpu).get()
    b = A.dot(A)
    
    assert np.allclose(b_gpu, b)

    
    

# fname = '../data/eq_test.h5'
# test_eqs(fname)

test_hardware()

