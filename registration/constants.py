import numpy as np

class TpsConstant(object):
    N_ITER        = 20
    EM_ITER       = 2
    REG           = (10, .1)
    RAD           = (.04, .00004)
    ROT_REG       = np.r_[1e-4, 1e-4, 1e-1]
    OUTLIER_PRIOR = .1
    OURLIER_FRAC  = 1e-2
