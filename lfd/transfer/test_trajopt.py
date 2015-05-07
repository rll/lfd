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

def base_pose_to_mat(traj, z):
    result = np.zeros((len(traj), 4, 4))
    for i in range(len(traj)):
        pose = traj[i]
        x, y, rot = pose
        q = openravepy.quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, z]
        mat = openravepy.matrixFromPose(q + pos)
        # import pdb; pdb.set_trace()
        result[i,:,:] = mat
    return result

def mat_to_base_pose(traj):
    """
    Untested
    """
    result = np.zeroes((len(traj), 3))
    for i in range(len(traj)):
        mat = traj[i]
        pose = openravepy.poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        rot = openravepy.axisAngleFromRotationMatrix(mat)[2]
        result[i,:] = np.array([x, y, rot])
    return result
        
def plan_trajopt(env, robot, target_pose, plotting=False):
    """
    Trajectory transfer of demonstrations using dual decomposition incorporating feedback
    """
    ### TODO: Need to tune parameters !!
    gamma = 1000.0
    print 'gamma = ', gamma # only gamma in use (for cost of trajectory)

    dim = 2

    # Get openrave robot
    rave_robot = env.sim.env.GetRobots()[0]
    z = rave_robot.GetTransform()[2, 3]
    import pdb; pdb.set_trace()


    ####### Figure out trajopt call ##########
    start_fixed = True
    if start_fixed:
        pass # do something

    n_steps = 15
    # manip_name = "base"
    # maintain_rel = True
    # maintain_up = True
    # collisionfree = True
    # init_traj = None
    # active_bodyparts = None #??
    # bodypart_init_dofs= {'base': target_pose[:,3][:3]} ### What should the DOF value be here???
    # assert type(target_pose) == list

    target_pose_7 = openravepy.poseFromMatrix(target_pose).tolist()

    # modify angle to be within +/- pi (not needed for now)
    # rot_i = self.robot().GetAffineDOFIndex(openravepy.DOFAffine.RotationAxis)
    # robot_z = self.robot().GetActiveDOFValues()[rot_i]
    # dofs[-1] = closer_ang(dofs[-1], robot_z)


    ################### Set up trajopt call
    ### Base motion planning must have init dofs??
    costs = []
    constraints = []
    # cost_coeffs = len(init_dofs) * [50]
    # cost_coeffs[-1] = 500
    # cost_coeffs[-2] = 500
    # cost_coeffs[-3] = 500

    # joint velocity cost
    joint_vel_cost = {
        "type": "joint_vel",
        "params": {"coeffs": [gamma/(n_steps - 1)]}
    }
    costs.append(joint_vel_cost)

    base_link = "base"

    constraints.append({
        "type": "pose",
        "params": {"xyz": target_pose_7[4:],
                   "wxyz": target_pose_7[:4],
                   "link": base_link,
                   "pos_coeffs": [20, 20, 20],
                   "rot_coeffs": [20, 20, 20]}
    })
                    

    request = {
        "basic_info": {
          "n_steps": n_steps,
          "manip": "base",
          "start_fixed": True  # i.e., DOF values at first timestep are fixed based on current robot state
        },
        "costs": costs,
        "constraints": constraints,
        "init_info":  {
            "type": "stationary"
        }
    }

    penalty_coeffs = [3000]
    dist_pen = [0.025]

    request['costs'] += [{
        "type": "collision",
        "name": "col",
        "params": {
          "continuous": True,
          # "coeffs": [20],
          "coeffs": [7000],
          "dist_pen": [0.02]
        }
    }]

    prob = trajoptpy.ConstructProblem(json.dumps(request), env.sim.env)

    result = trajoptpy.OptimizeProblem(prob)
    traj = result.GetTraj()
    import pdb; pdb.set_trace()
    total_cost = sum(cost[1] for cost in result.GetCosts())
    traj_mat = base_pose_to_mat(traj, z)
    

    return traj_mat
