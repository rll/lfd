JOINT_LENGTH_PER_STEP = .1
FINGER_CLOSE_RATE     = .1

BETA_POS = 1000000.0
BETA_ROT = 100.0
GAMMA = 1000.0
USE_COLLISION_COST = True

try:
	from lfd_settings.transfer.settings import *
except ImportError:
	pass
