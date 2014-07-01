import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from scikits.cuda import misc, cublas
from string import lower, upper
import numpy as np

def get_gpu_ptrs(arr, (m, n) = (0, 0)):
    shape = arr[0].shape
    if len(shape) == 1:        
        ptrs = [int(x[m:].gpudata) for x in arr]
    else:
        ptrs = [int(x[m:, n:].gpudata) for x in arr]
    return gpuarray.to_gpu(np.array(ptrs))

def m_dot_batch(*args):
    tmp_arr, tmp_ptrs, _trans = args[0]
    for next_arr, next_ptrs, trans in args[1:]:
        tmp_arr, tmp_ptrs = dot_batch(tmp_arr, next_arr, tmp_ptrs, next_ptrs,
                                      transa = _trans, transb = trans)
        _trans = 'N'#so we only do it once
    return tmp_arr, tmp_ptrs
    
def dot_batch(a_arr_gpu, b_arr_gpu, a_ptr_gpu, b_ptr_gpu,
              transa = 'N', transb = 'N'):
    N = len(a_arr_gpu)
    a_shape = a_arr_gpu[0].shape
    a_dtype = a_arr_gpu[0].dtype
    if len(a_shape) == 1:        
        a_shape = (1, a_shape[0])
    b_shape = b_arr_gpu[0].shape
    if len(b_shape) == 1:
        b_shape = (1, b_shape[0])

    transa = lower(transa)
    transb = lower(transb)

    if transb in ['t', 'c']:
        m, k = b_shape
    elif transb in ['n']:
        k, m = b_shape
    else:
        raise ValueError('invalid value for transb')
    
    if transa in ['t', 'c']:
        l, n = a_shape
    elif transa in ['n']:
        n, l = a_shape
    else:
        raise ValueError('invalid value for transa')    

    c_shape = (n, m)
    
    c_arr_gpu = [gpuarray.empty(c_shape, a_dtype) for _ in range(N)]
    c_ptr_gpu = get_gpu_ptrs(c_arr_gpu)

    dot_batch_nocheck(a_arr_gpu, b_arr_gpu, c_arr_gpu,
                      a_ptr_gpu, b_ptr_gpu, c_ptr_gpu,
                      transa, transb, b = 0)
    return c_arr_gpu, c_ptr_gpu

def dot_batch_nocheck(a_arr_gpu, b_arr_gpu, c_arr_gpu, a_ptr_gpu, b_ptr_gpu, c_ptr_gpu,
                      transa = 'N', transb = 'N', a = 1, b = 1, handle = None):
    """
    Implementation of batched dot products using cuda.    

    Parameters
    ----------
    a_arr_gpu : list of pycuda.gpuarray.GPUArray
        Input array.
    b_arr_gpu : list of pycuda.gpuarray.GPUArray
        Input array.
    c_arr_gpu : list of pycuda.gpuarray.GPUArray
        Input/Output array.
    a_ptr_gpu : pycuda.gpuarray.GPUArray
        Array of pointers to arrays in a_arr_gpu
    b_ptr_gpu : pycuda.gpuarray.GPUArray
        Array of pointers to arrays in b_arr_gpu
    c_ptr_gpu : pycuda.gpuarray.GPUArray
        Array of pointers to arrays in c_arr_gpu
    transa : char
        If 'T', compute the product of the transpose of `a_arr_gpu[i]`.
        If 'C', compute the product of the Hermitian of `a_arr_gpu[i]`.
    transb : char
        If 'T', compute the product of the transpose of `b_arr_gpu[i]`.
        If 'C', compute the product of the Hermitian of `b_arr_gpu[i]`.
    handle : int
        CUBLAS context. If no context is specified, the default handle from
        `scikits.cuda.misc._global_cublas_handle` is used.

    Returns
    -------
    None. Output is stored into the arrays in c_arr_gpu

    Notes
    -----
    The input matrices must all contain elements of the same data type.
    All matrics in a list must have same size.

    Examples
    --------
    >>> import pycuda.driver
    >>> import pycuda.autoinit
    >>> from pycuda import gpuarray
    >>> import numpy as np
    >>> from scikits.cuda import linalg
    >>> linalg.init()
    >>> a_arr = [np.asarray(np.random.rand(4, 2), np.float32) for i in range(10)]
    >>> b_arr = [np.asarray(np.random.rand(2, 2), np.float32) for i in range(10)]
    >>> c_arr = [np.asarray(np.random.rand(4, 2), np.float32) for i in range(10)]
    >>> a_arr_gpu = [gpuarray.to_gpu(a_gpu) for a_gpu in a_arr]
    >>> b_arr_gpu = [gpuarray.to_gpu(b_gpu) for b_gpu in b_arr]
    >>> c_arr_gpu = [gpuarray.to_gpu(c_gpu) for c_gpu in c_arr]
    >>> a_ptr_gpu = gpuarray.to_gpu(np.asarray([int(a_gpu.gpudata) for a_gpu in a_arr_gpu]))    
    >>> b_ptr_gpu = gpuarray.to_gpu(np.asarray([int(b_gpu.gpudata) for b_gpu in b_arr_gpu]))
    >>> c_ptr_gpu = gpuarray.to_gpu(np.asarray([int(c_gpu.gpudata) for c_gpu in c_arr_gpu]))
    >>> linalg.dot_batch_nocheck(a_arr_gpu, b_arr_gpu, c_arr_gpu, a_ptr_gpu, b_ptr_gpu, c_ptr_gpu)
    >>> for i in range(10):
    ...   print np.allclose(np.dot(a_arr[i], b_arr[i]) + c_arr[i], c_arr_gpu[i].get())
    ... 
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    >>>
    """
    if handle is None:
        handle = misc._global_cublas_handle
    N = len(a_arr_gpu)
    a_shape = a_arr_gpu[0].shape
    a_dtype = a_arr_gpu[0].dtype
    b_shape = b_arr_gpu[0].shape
    b_dtype = b_arr_gpu[0].dtype
    c_shape = c_arr_gpu[0].shape
    c_dtype = c_arr_gpu[0].dtype
    
    for i in range(N):
        assert a_arr_gpu[i].shape == a_shape
        assert a_arr_gpu[i].dtype == a_dtype
        assert b_arr_gpu[i].shape == b_shape
        assert b_arr_gpu[i].dtype == b_dtype
        assert c_arr_gpu[i].shape == c_shape
        assert c_arr_gpu[i].dtype == c_dtype

        assert a_arr_gpu[i].flags.c_contiguous
        assert b_arr_gpu[i].flags.c_contiguous
        assert c_arr_gpu[i].flags.c_contiguous

    if len(a_shape) == 1:        
        a_shape = (1, a_shape[0])
    if len(b_shape) == 1:
        b_shape = (1, b_shape[0])
    if len(c_shape) == 1:
        c_shape = (1, c_shape[0])

    transa = lower(transa)
    transb = lower(transb)

    if transb in ['t', 'c']:
        m, k = b_shape
    elif transb in ['n']:
        k, m = b_shape
    else:
        raise ValueError('invalid value for transb')
    
    if transa in ['t', 'c']:
        l, n = a_shape
    elif transa in ['n']:
        n, l = a_shape
    else:
        raise ValueError('invalid value for transa')

    i, j = c_shape
    
    if l != k:
        raise ValueError('objects are not aligned')    
    if i != n:
        raise ValueError('objects are not aligned')
    if j != m:
        raise ValueError('objects are not aligned')
    
    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)
    
    if (a_dtype == np.complex64 and b_dtype == np.complex64 \
            and c_dtype == np.complex64):
        cublas_func = cublas.cublasCgemmBatched
        alpha = np.complex64(a)
        beta = np.complex64(b)
    elif (a_dtype == np.float32 and b_dtype == np.float32\
            and c_dtype == np.float32):
        cublas_func = cublas.cublasSgemmBatched
        alpha = np.float32(a)
        beta = np.float32(b)
    elif (a_dtype == np.complex128 and b_dtype == np.complex128\
            and c_dtype == np.complex128):
        cublas_func = cublas.cublasZgemmBatched
        alpha = np.complex128(a)
        beta = np.complex128(b)
    elif (a_dtype == np.float64 and b_dtype == np.float64\
            and c_dtype == np.float64):
        cublas_func = cublas.cublasDgemmBatched
        alpha = np.float64(a)
        beta = np.float64(b)
    else:
        raise ValueError('unsupported combination of input types')

    cublas_func(handle, transb, transa, m, n, k, alpha, b_ptr_gpu.gpudata, lda, 
                a_ptr_gpu.gpudata, ldb, beta, c_ptr_gpu.gpudata, ldc, N)
# @profile
def batch_sum(a_arr_gpu, a_ptr_gpu):
    """
    computes a sum of all of the arrays pointed to by a_arr_gpu and a_ptr_gpu
    """
    if len(a_arr_gpu[0].shape) != 1:
        n, m       = a_arr_gpu[0].shape
        total_size = n * m
        flat_a_gpu = [a.ravel() for a in a_arr_gpu]
    else:
        total_size = a_arr_gpu[0].shape[0]
        flat_a_gpu = a_arr_gpu

    ones_vec      = gpuarray.to_gpu_async(np.ones((total_size, 1), dtype=np.float32))
    ones_arr_gpu  = [ones_vec for i in range(len(a_arr_gpu))]
    ones_ptr_gpu  = get_gpu_ptrs(ones_arr_gpu)

    res_arr, res_ptrs = dot_batch(flat_a_gpu, ones_arr_gpu, a_ptr_gpu, ones_ptr_gpu)
    return [r.get()[0] for r in res_arr]
    
## cribbed from
## http://hannes-brt.github.io/blog/2013/08/07/column-slicing-in-py
def extract_cols(mat, start=0, stop=None):
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    N, M = mat.shape
    m = stop - start

    assert mat.flags.c_contiguous
    assert start >= 0 and start <= M and stop >= 0 and stop <= M and stop > start

    new_mat = gpuarray.empty((N, m), dtype)

    copy = drv.Memcpy2D()
    copy.set_src_device(mat.gpudata)
    copy.src_x_in_bytes = start * itemsize    # Offset of the first column in bytes
    copy.set_dst_device(new_mat.gpudata)
    copy.src_pitch = M * itemsize   # Width of a row in bytes in the source array
    copy.dst_pitch = copy.width_in_bytes = m * itemsize  # Width of sliced row
    copy.height = N
    copy(aligned=True)

    return new_mat

if __name__ == '__main__':
    import pycuda.autoinit
    import numpy as np
    from scikits.cuda import linalg
    linalg.init()
    N = 100
    a_arr = [np.asarray(np.random.rand(4, 2), np.float32) for _ in range(N)]
    b_arr = [np.asarray(np.random.rand(2, 2), np.float32) for _ in range(N)]
    c_arr = [np.asarray(np.random.rand(4, 2), np.float32) for _ in range(N)]
    a_arr_gpu = [gpuarray.to_gpu(a_gpu) for a_gpu in a_arr]
    b_arr_gpu = [gpuarray.to_gpu(b_gpu) for b_gpu in b_arr]
    c_arr_gpu = [gpuarray.to_gpu(c_gpu) for c_gpu in c_arr]
    a_ptr_gpu = gpuarray.to_gpu(np.asarray([int(a_gpu.gpudata) for a_gpu in a_arr_gpu]))    
    b_ptr_gpu = gpuarray.to_gpu(np.asarray([int(b_gpu.gpudata) for b_gpu in b_arr_gpu]))
    c_ptr_gpu = gpuarray.to_gpu(np.asarray([int(c_gpu.gpudata) for c_gpu in c_arr_gpu]))
    dot_batch_nocheck(a_arr_gpu, b_arr_gpu, c_arr_gpu, a_ptr_gpu, b_ptr_gpu, c_ptr_gpu)
    success = True
    for i in range(N):
       success &= np.allclose(np.dot(a_arr[i], b_arr[i]) + c_arr[i], c_arr_gpu[i].get())
    print "Test Successful:\t{}".format(success)
