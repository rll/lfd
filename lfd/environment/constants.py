class GripperConstant(object):
    OPEN_CLOSE_THRESH = 0.04 # .06 for fig8 (thick rope), 0.04 for thin rope (overhand)

class RopeConstant(object):
    RADIUS        = .005
    ANG_STIFFNESS = .1
    ANG_DAMPING   = 1
    LIN_DAMPING   = .75
    ANG_LIMIT     = .4
    LIN_STOP_ERP  = .2
    MASS          = 1.0
    RADIUS_THICK  = .008
