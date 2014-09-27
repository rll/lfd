from tpsopt.constants import *
import numpy as np

GRIPPER_OPEN_CLOSE_THRESH = 0.04 # .07 for thick rope, but buggy??
COLLISION_DIST_THRESHOLD = 0.0
MAX_ACTIONS_TO_TRY = 10  # Number of actions to try (ranked by cost), if TrajOpt trajectory is infeasible
TRAJOPT_MAX_ACTIONS = 5  # Number of actions to compute full feature (TPS + TrajOpt) on
WEIGHTS = np.array([-1]) 
#DS_SIZE = .03
DS_SIZE = .025
##################
#  ROPE DEFAULTS
##################
ROPE_RADIUS        = .005
ROPE_ANG_STIFFNESS = .1
ROPE_ANG_DAMPING   = 1
ROPE_LIN_DAMPING   = .75
ROPE_ANG_LIMIT     = .4
ROPE_LIN_STOP_ERP  = .2
ROPE_MASS          = 1.0
ROPE_RADIUS_THICK  = .008

##################
# Planning Params
##################
JOINT_LENGTH_PER_STEP = .1
FINGER_CLOSE_RATE     = .1
