import copy
import datetime
import settings
import json
import trajoptpy
import openravepy
import numpy as np
import sys
from lfd.demonstration import demonstration
from lfd.environment import sim_util
from lfd.registration import registration, tps
from lfd.transfer import transfer
from lfd.transfer import planning
from lfd.util import util


def plan_trajopt(env, robot, plotting=False):
    """
    Trajectory transfer of demonstrations using dual decomposition incorporating feedback
    """
    ### TODO: Need to tune parameters !!
    gamma = 1000.0
    print 'gamma = ', gamma # only gamma in use (for cost of trajectory)

    dim = 2


    ####### Figure out trajopt call ##########
    start_fixed = True
    if start_fixed:
        pass # do something
    n_steps = demo_traj.shape[0]
    manip_name = "base"

    request = {
        "basic_info": {
            "n_steps": n_steps,
            "manip_name": manip_name,
            "start_fixed": start_fixed
        },
        "costs": [
        { 
            "type": "joint_vel",
            "params": {"coeffs": [self.gamma/(n_steps - 1)]}
        },
        ],
        "constraints" : [
        ],
    }

    penalty_coeffs = [3000]
    dist_pen = [0.025]
    if self.use_collision_cost:
        ## append collision cost
        requests["costs"].append(
            {
                "type": "collision",
                "params": {
                    "continuous" : True,
                    "coeffs" : penalty_coeffs, 
                    "dist_pen" : dist_pen
                }
            })

    
    ##### To be implemented #####
    ############### Update Dual variables ###############


