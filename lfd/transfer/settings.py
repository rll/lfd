#:
JOINT_LENGTH_PER_STEP = .1
#:
FINGER_CLOSE_RATE     = .1

#: TPS objective coefficient in unified trajectory optimization
ALPHA = 1000000.0
#: pose position or finger position objective coefficient in trajectory optimization
BETA_POS = 1000000.0
#: pose rotation objective coefficient in trajectory optimization
BETA_ROT = 100.0
#: joint velocity objective coefficient in trajectory optimization
GAMMA = 1000.0
#: whether to use collision cost in trajectory optimization
USE_COLLISION_COST = True

try:
	from lfd_settings.transfer.settings import *
except ImportError:
	pass
