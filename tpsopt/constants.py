N_ITER_CHEAP       = 10
N_ITER_EXACT       = 50
EM_ITER_CHEAP      = 1
DEFAULT_LAMBDA     = (.1, .001)
#needs to line up with the MAX_DIM from tps.cuh
from cuda_funcs import get_max_cld_size
MAX_CLD_SIZE       = get_max_cld_size()
MAX_TRAJ_LEN       = 100
EXACT_LAMBDA       = (10, .001)
DATA_DIM           = 3
DS_SIZE            = 0.025
N_STREAMS          = 10
DEFAULT_NORM_ITERS = 10
BEND_COEF_DIGITS   = 6
