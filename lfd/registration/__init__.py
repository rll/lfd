try:
    # initialize pycuda
    import pycuda.autoinit
    
    # initialize libraries used by scikits.cuda
    import scikits.cuda.linalg
    scikits.cuda.linalg.init()
    
    _has_cuda = True
    _has_cula = scikits.cuda.linalg._has_cula
except (ImportError, OSError):
    _has_cuda = False
    _has_cula = False
