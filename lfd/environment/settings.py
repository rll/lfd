GRIPPER_OPEN_CLOSE_THRESH = 0.04 # .06 for fig8 (thick rope), 0.04 for thin rope (overhand)

ROPE_RADIUS        = .005
ROPE_ANG_STIFFNESS = .1
ROPE_ANG_DAMPING   = 1
ROPE_LIN_DAMPING   = .75
ROPE_ANG_LIMIT     = .4
ROPE_LIN_STOP_ERP  = .2
ROPE_MASS          = 1.0
ROPE_RADIUS_THICK  = .008

try:
	from lfd_settings.environment.settings import *
except ImportError:
	pass
