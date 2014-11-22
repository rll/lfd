#:
GRIPPER_OPEN_CLOSE_THRESH = 0.04 # .06 for fig8 (thick rope), 0.04 for thin rope (overhand)

#:
ROPE_RADIUS        = .005
#:
ROPE_ANG_STIFFNESS = .1
#:
ROPE_ANG_DAMPING   = 1
#:
ROPE_LIN_DAMPING   = .75
#:
ROPE_ANG_LIMIT     = .4
#:
ROPE_LIN_STOP_ERP  = .2
#:
ROPE_MASS          = 1.0
#:
ROPE_RADIUS_THICK  = .008

#: window properties for the viewer's window
WINDOW_PROP = [0,0,1500,1500]
#: transposed homogeneous matrix for the viewer's camera
CAMERA_MATRIX = [[    0, 1,   0, 0],
				 [   -1, 0, 0.5, 0],
				 [  0.5, 0,   1, 0],
				 [ 2.25, 0, 4.5, 1]]

try:
	from lfd_settings.environment.settings import *
except ImportError:
	pass
