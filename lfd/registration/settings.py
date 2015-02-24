import numpy as np

# RPM registration
#: number of outer iterations for RPM algorithms
N_ITER        = 20
#: number of inner iterations for RPM algorithms
EM_ITER       = 1
#: initial and final smoothing regularizer coefficient for RPM algorithms
REG           = (.1, .0001)
#: initial and final temperature for RPM algorithms
RAD           = (.01, .0001)
#: rotation regularizer coefficients
ROT_REG       = np.r_[1e-4, 1e-4, 1e-1]
#:
OUTLIER_PRIOR = .1
#:
OUTLIER_FRAC  = 1e-2


# registration with gpu
#:
MAX_CLD_SIZE       = 150
#:
BEND_COEF_DIGITS   = 6
#:
OUTLIER_CUTOFF  = 1e-2


# L2 registration
#:
L2_N_ITER   = 4
#:
L2_OPT_ITER = 100
#:
L2_REG      = (.1, .01)
#:
L2_RAD      = (.1, .01)
#:
L2_ROT_REG  = np.r_[1e-4, 1e-4, 1e-1]
#:


# multi registration
#:
COV_COEF = .1

try:
	from lfd_settings.registration.settings import *
except ImportError:
	pass
