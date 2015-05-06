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
        
def plan_trajopt_test_linear(env, robot, rel_pts, demo, target_pose, plotting=False):
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


    ####### Figure out trajopt call ##########
    start_fixed = True
    if start_fixed:
        pass # do something

    ######
    # n_steps = 15
    n_steps = len(demo.scene_states)

    target_pose_7 = openravepy.poseFromMatrix(target_pose).tolist()

    # modify angle to be within +/- pi (not needed for now)
    # rot_i = self.robot().GetAffineDOFIndex(openravepy.DOFAffine.RotationAxis)
    # robot_z = self.robot().GetActiveDOFValues()[rot_i]
    # dofs[-1] = closer_ang(dofs[-1], robot_z)


    ################### Set up trajopt call ##########
    costs = []
    constraints = []

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


    ############################### ABOVE WORKS ################################
    ############### Include the linear terms in the objective ##################

    demo_pc_seq = demo.scene_states
    demo_traj = demo.traj

    dim = 2

    # How many time steps of all the demos to actually use (0, timestep_dist, 2*timestep_dist, ...)
    timestep_dist = 5
    num_time_steps = len(demo_pc_seq)
    assert type(timestep_dist) == int
    time_step_indices = np.arange(0, num_time_steps, timestep_dist)
    demo_pc_seq = demo_pc_seq[time_step_indices]
    # demo_traj = demo_traj[time_step_indices]

    # dual variable for sequence of point clouds
    points_per_pc = len(demo_pc_seq[0])
    num_time_steps = len(demo_pc_seq)
    total_pc_points = points_per_pc * num_time_steps
    lamb = np.zeros((total_pc_points, dim)) # dual variable for point cloud points (1260 x 3)

    # convert the dimension of point cloud
    demo_pc = demo_pc_seq.reshape((total_pc_points, 3)) 
    demo_pc = demo_pc[:,:dim]

    # convert trajectory to points 
    orig_demo_pc_seq = demo.scene_states


    #####################################
    # convert trajectory to points 
    rel_pts_traj_seq = demo.rel_pts_traj_seq
    ########### REDUCE THE NUMBER OF POINTS FOR TRAJECTORY SAMPLE
    demo_traj_pts = rel_pts_traj_seq #self.traj_to_points() # simple case: (TODO) implement the complicated version

    # dual variable for sequence of trajectories
    points_per_traj = len(demo_traj_pts[0])
    num_time_steps = len(demo_traj_pts)
    total_traj_points = points_per_traj * num_time_steps
    nu_bd = np.zeros((total_traj_points, dim))
   
    # convert the dimension of tau_bd
    tau_bd = demo_traj_pts.reshape((total_traj_points, 3))
    tau_bd = tau_bd[:,:dim]
    
    # (TODO) Need to convert tau to relative points (ignore for now)
    
    ###### In summary ######
    # demo_pc: #number of sampled timesteps x points_per_pc
    # lamb: #number of sampled timesteps x points_per_pc
    # tau_bd: #total_number of trajectory points x points_per_trajectory
    # nu_bd: #total_number of trajectory points x points_per_trajectory


    start_fixed = True
    lin_traj_coeff = 10000.0 ### what should i set this to ??
    ## Add linear term for trajectory (IN WHAT FRAME SHOULD THE POINTS BE IN)
    ## Should include every single step of the trajectory
    ## confusing part??
    # robot_rel_pts = 

    #### Use rel_pts here 
    #### 
    #### MAKE IT ONLY HAVE FOUR POINTS DURING INPUT (so that it doesn't run forever)
    #### input to be fixed
    assert num_time_steps == n_steps
    assert points_per_traj == len(rel_pts) 
    # import pdb; pdb.set_trace()
    nu_bd_trajopt = np.hstack((nu_bd, np.zeros((len(nu_bd), 1))))
    for i in range(num_time_steps):
        request['costs'].append(
            {"type":"rel_pts_nus",
                "params":{
                    "nus": (-nu_bd_trajopt[i * points_per_traj: (i+1) * points_per_traj]).tolist(),
                    "rel_xyzs": rel_pts.tolist(),
                    "link": base_link,
                    "timestep": i,
                    "pos_coeffs":[lin_traj_coeff / n_steps] * 4
                }
            }
        )
    
    lin_pc_coeff = 10000.0 #### TO BE TUNED
    ## Add linear term for pointcloud (more work to be done here: depends on the trajectory)
    ## Should include only the steps we would like to use to include the point clouds (every several time steps)
    ## How many do we need to use here? (the sequence of point clouds)
    num_pc_considered = len(time_step_indices)
    lamb_trajopt = np.hstack((lamb, np.zeros((len(lamb), 1))))

    ########### orig pc need not to be centered at 0.0

    ##### Use dimension of 3 here, below does not throw an error ####
    for i in range(num_pc_considered):
        timestep = time_step_indices[i]
        request['costs'].append(
            {"type": "pc_pts_lambdas",
                "params": {
                    "lambdas": (-lamb_trajopt[i * points_per_pc: (i+1) * points_per_pc]).tolist(),
                    "orig_pc": orig_demo_pc_seq[0].tolist(),
                    "link": base_link,
                    "timestep": timestep,
                    # "num_pc_considered": num_pc_considered,
                    # "pc_time_steps": time_step_indices,
                    "pos_coeffs":[lin_pc_coeff / num_pc_considered]
                }
            }
        )

    import pdb; pdb.set_trace()
    prob = trajoptpy.ConstructProblem(json.dumps(request), env.sim.env)

    result = trajoptpy.OptimizeProblem(prob)
    traj = result.GetTraj()
    import pdb; pdb.set_trace()
    total_cost = sum(cost[1] for cost in result.GetCosts())
    traj_mat = base_pose_to_mat(traj, z)

    return traj_mat
