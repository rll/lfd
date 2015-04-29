from __future__ import division

import copy
import numpy as np
import json
import openravepy
import trajoptpy
from lfd.environment import sim_util
from lfd.util import util
from lfd.rapprentice import math_utils as mu
import settings
import IPython as ipy

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj,
                     no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                     beta_pos=settings.BETA_POS, beta_rot=settings.BETA_ROT, gamma=settings.GAMMA):
    return plan_follow_trajs(robot, manip_name, [ee_link.GetName()], [new_hmats], old_traj,
                     no_collision_cost_first=no_collision_cost_first, use_collision_cost=use_collision_cost, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                     beta_pos=beta_pos, beta_rot=beta_rot, gamma=gamma)

def plan_follow_trajs(robot, manip_name, ee_link_names, ee_trajs, old_traj,
                     no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                     beta_pos=settings.BETA_POS, beta_rot=settings.BETA_ROT, gamma=settings.GAMMA):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()

    n_steps = len(ee_trajs[0])
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == len(dof_inds)
    assert len(ee_link_names) == len(ee_trajs)

    if no_collision_cost_first:
        init_traj, _, _ = plan_follow_trajs(robot, manip_name, ee_link_names, ee_trajs, old_traj,
                                        no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                        beta_pos = beta_pos, beta_rot = beta_rot, gamma = gamma)
    else:
        init_traj = old_traj.copy()

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }

    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })

    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    for (ee_link_name, ee_traj) in zip(ee_link_names, ee_trajs):
        poses = [openravepy.poseFromMatrix(hmat) for hmat in ee_traj]
        for (i_step,pose) in enumerate(poses):
            if start_fixed and i_step == 0:
                continue
            request["costs"].append(
                {"type":"pose",
                 "params":{
                    "xyz":pose[4:7].tolist(),
                    "wxyz":pose[0:4].tolist(),
                    "link":ee_link_name,
                    "timestep":i_step,
                    "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*3,
                    "rot_coeffs":[np.sqrt(beta_rot/n_steps)]*3
                 }
                })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        orig_dof_vals
        with util.suppress_stdout():
          prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
          result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()

    pose_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "pose"])
    pose_err = []
    with openravepy.RobotStateSaver(robot):
        for (ee_link_name, ee_traj) in zip(ee_link_names, ee_trajs):
            ee_link = robot.GetLink(ee_link_name)
            for (i_step, hmat) in enumerate(ee_traj):
                if start_fixed and i_step == 0:
                    continue
                robot.SetDOFValues(traj[i_step], dof_inds)
                new_hmat = ee_link.GetTransform()
                pose_err.append(openravepy.poseFromMatrix(mu.invertHmat(hmat).dot(new_hmat)))
    pose_err = np.asarray(pose_err)
    pose_costs2 = (beta_rot/n_steps) * np.square(pose_err[:,1:4]).sum() + (beta_pos/n_steps) * np.square(pose_err[:,4:7]).sum()

    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/(n_steps-1)) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])

    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("pose(s)", pose_costs, pose_costs2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rot pose(s)", np.abs(pose_err[:,1:4]).min(), np.abs(pose_err[:,1:4]).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("trans pose(s)", np.abs(pose_err[:,4:7]).min(), np.abs(pose_err[:,4:7]).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())

    return traj, obj_value, pose_costs

def plan_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2finger_pts_traj, old_traj,
                                no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                beta_pos=settings.BETA_POS, gamma=settings.GAMMA):
    return plan_follow_finger_pts_trajs(robot, manip_name, [flr2finger_link.GetName()], flr2finger_rel_pts, [flr2finger_pts_traj], old_traj,
                                no_collision_cost_first=no_collision_cost_first, use_collision_cost=use_collision_cost, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                beta_pos=beta_pos, gamma=gamma)

def plan_follow_finger_pts_trajs(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2finger_pts_trajs, old_traj,
                                no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                beta_pos=settings.BETA_POS, gamma=settings.GAMMA):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()

    n_steps = old_traj.shape[0]
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[1] == len(dof_inds)
    for flr2finger_pts_traj in flr2finger_pts_trajs:
        for finger_pts_traj in flr2finger_pts_traj.values():
            assert len(finger_pts_traj)== n_steps
    assert len(flr2finger_link_names) == len(flr2finger_pts_trajs)

    if no_collision_cost_first:
        init_traj, _ = plan_follow_finger_pts_trajs(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2finger_pts_trajs, old_traj,
                                                   no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                                   beta_pos = beta_pos, gamma = gamma)
    else:
        init_traj = old_traj.copy()

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]


    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }

    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [5000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.1]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })

    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    # This is the finger following constraint
    for (flr2finger_link_name, flr2finger_pts_traj) in zip(flr2finger_link_names, flr2finger_pts_trajs):
        for finger_lr, finger_link_name in flr2finger_link_name.items():
            finger_rel_pts = flr2finger_rel_pts[finger_lr]
            finger_pts_traj = flr2finger_pts_traj[finger_lr]
            for (i_step, finger_pts) in enumerate(finger_pts_traj):
                if start_fixed and i_step == 0:
                    continue
                request["costs"].append(
                    {"type":"rel_pts",
                     "params":{
                        "xyzs":finger_pts.tolist(),
                        "rel_xyzs":finger_rel_pts.tolist(),
                        "link":finger_link_name,
                        "timestep":i_step,
                        "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4, # there is a coefficient for each of the 4 points
                     }
                    })

    # Penalize the x direction on right gripper for 36 timesteps
    #penalty = np.zeros([4,3])
    #penalty[:,1] = -100000
    #for i in range(36):
      #request["costs"].append(
         #{"type":"rel_pts_lambdas",
           #"params":{
             #"lambdas":penalty.tolist(),
             #"rel_xyzs":flr2finger_rel_pts['r'].tolist(),
             #"link":flr2finger_link_names[0]['r'],
             #"timestep":i,
            #}
         #})
    #import IPython as ipy; ipy.embed()
    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization

    traj = result.GetTraj()
    #ipy.embed()

    rel_pts_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "rel_pts"])
    rel_pts_err = []
    with openravepy.RobotStateSaver(robot):
        for (flr2finger_link_name, flr2finger_pts_traj) in zip(flr2finger_link_names, flr2finger_pts_trajs):
            for finger_lr, finger_link_name in flr2finger_link_name.items():
                finger_link = robot.GetLink(finger_link_name)
                finger_rel_pts = flr2finger_rel_pts[finger_lr]
                finger_pts_traj = flr2finger_pts_traj[finger_lr]
                for (i_step, finger_pts) in enumerate(finger_pts_traj):
                    if start_fixed and i_step == 0:
                        continue
                    robot.SetDOFValues(traj[i_step], dof_inds)
                    new_hmat = finger_link.GetTransform()
                    rel_pts_err.append(finger_pts - (new_hmat[:3,3][None,:] + finger_rel_pts.dot(new_hmat[:3,:3].T)))
    rel_pts_err = np.concatenate(rel_pts_err, axis=0)
    rel_pts_costs2 = (beta_pos/n_steps) * np.square(rel_pts_err).sum() # TODO don't square n_steps

    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/(n_steps-1)) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])

    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rel_pts(s)", rel_pts_costs, rel_pts_costs2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rel_pts(s)", np.abs(rel_pts_err).min(), np.abs(rel_pts_err).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
    return traj, obj_value, rel_pts_costs

def joint_fit_tps_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2finger_pts_traj, old_traj,
                                              f, closing_pts=None,
                                              no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                              alpha=settings.ALPHA, beta_pos=settings.BETA_POS, gamma=settings.GAMMA):
    return joint_fit_tps_follow_finger_pts_trajs(robot, manip_name, [flr2finger_link.GetName()], flr2finger_rel_pts, [flr2finger_pts_traj], old_traj,
                                                 f, closing_pts=closing_pts,
                                                 no_collision_cost_first=no_collision_cost_first, use_collision_cost=use_collision_cost, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                                 alpha=alpha, beta_pos=beta_pos, gamma=gamma)

def joint_fit_tps_follow_finger_pts_trajs(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2old_finger_pts_trajs, old_traj,
                                         f, closing_pts=None,
                                         no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                          alpha=settings.ALPHA, beta_pos=settings.BETA_POS, gamma=settings.GAMMA):
    # jointly optimize over registration and trajectory
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()

    # some sanity checks
    n_steps = old_traj.shape[0]
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[1] == len(dof_inds)
    for flr2old_finger_pts_traj in flr2old_finger_pts_trajs:
        for old_finger_pts_traj in flr2old_finger_pts_traj.values():
            assert len(old_finger_pts_traj)== n_steps
    assert len(flr2finger_link_names) == len(flr2old_finger_pts_trajs)

    # expand these 
    (n,d) = f.x_na.shape
    bend_coefs = np.ones(d) * f.bend_coef if np.isscalar(f.bend_coef) else f.bend_coef
    rot_coefs = np.ones(d) * f.rot_coef if np.isscalar(f.rot_coef) else f.rot_coef
    if f.wt_n is None:
        wt_n = np.ones(n)
    else:
        wt_n = f.wt_n
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))

    if no_collision_cost_first:
        # never used?
        import pdb; pdb.set_trace()
        init_traj, _, (N, init_z) , _, _ = joint_fit_tps_follow_finger_pts_trajs(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2old_finger_pts_trajs, old_traj,
                                                                                 f, closing_pts=closing_pts,
                                                                                 no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                                                                 alpha=alpha, beta_pos=beta_pos, gamma=gamma)
    else:
        init_traj = old_traj.copy()
        N = f.N
        init_z = f.z


    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    #ipy.embed();
    request = {
        "traj_basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "basic_info" : {
            "n_steps" : n_steps,
            "m_ext" : n,
            "n_ext" : d,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        {
            "type" : "tps",
            "name" : "tps",
            "params" : {"x_na" : [row.tolist() for row in f.x_na],
                        "y_ng" : [row.tolist() for row in f.y_ng],
                        "bend_coefs" : bend_coefs.tolist(),
                        "rot_coefs" : rot_coefs.tolist(),
                        "wt_n" : [row.tolist() for row in wt_n],
                        "N" : [row.tolist() for row in N],
                        "alpha" : alpha,
            }
        }
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj],
            "data_ext":[row.tolist() for row in init_z]
        }
    }

    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })

    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    if closing_pts is not None:
        request["costs"].append(
            {
                "type":"tps_jac_orth",
                "params":  {
                            "tps_cost_name":"tps",
                            "pts":closing_pts.tolist(),
                            "coeffs":[10.0]*len(closing_pts),
                            }
            })

    for (flr2finger_link_name, flr2old_finger_pts_traj) in zip(flr2finger_link_names, flr2old_finger_pts_trajs):
        for finger_lr, finger_link_name in flr2finger_link_name.items():
            finger_rel_pts = flr2finger_rel_pts[finger_lr]
            old_finger_pts_traj = flr2old_finger_pts_traj[finger_lr]
            for (i_step, old_finger_pts) in enumerate(old_finger_pts_traj):
                if start_fixed and i_step == 0:
                    continue
                request["costs"].append(
                    {"type":"tps_rel_pts",
                     "params":{
                        "tps_cost_name":"tps",
                        "src_xyzs":old_finger_pts.tolist(),
                        "rel_xyzs":finger_rel_pts.tolist(),
                        "link":finger_link_name,
                        "timestep":i_step,
                        "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4,
                     }
                    })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
          prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
          result = trajoptpy.OptimizeProblem(prob) # do optimization

    traj = result.GetTraj()
    f.z = result.GetExt()
    theta = N.dot(f.z)
    f.trans_g = theta[0,:]
    f.lin_ag = theta[1:d+1,:]
    f.w_ng = theta[d+1:]

    tps_rel_pts_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "rel_pts"])
    tps_rel_pts_err = []
    with openravepy.RobotStateSaver(robot):
        for (flr2finger_link_name, flr2old_finger_pts_traj) in zip(flr2finger_link_names, flr2old_finger_pts_trajs):
            for finger_lr, finger_link_name in flr2finger_link_name.items():
                finger_link = robot.GetLink(finger_link_name)
                finger_rel_pts = flr2finger_rel_pts[finger_lr]
                old_finger_pts_traj = flr2old_finger_pts_traj[finger_lr]
                for (i_step, old_finger_pts) in enumerate(old_finger_pts_traj):
                    if start_fixed and i_step == 0:
                        continue
                    robot.SetDOFValues(traj[i_step], dof_inds)
                    new_hmat = finger_link.GetTransform()
                    tps_rel_pts_err.append(f.transform_points(old_finger_pts) - (new_hmat[:3,3][None,:] + finger_rel_pts.dot(new_hmat[:3,:3].T)))
    tps_rel_pts_err = np.concatenate(tps_rel_pts_err, axis=0)
    tps_rel_pts_costs2 = (beta_pos/n_steps) * np.square(tps_rel_pts_err).sum() # TODO don't square n_steps

    tps_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "tps"])
    tps_cost2 = alpha * f.get_objective().sum()
    matching_err = f.transform_points(f.x_na) - f.y_ng

    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/(n_steps-1)) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    tps_jac_orth_cost = [cost_val for (cost_type, cost_val) in result.GetCosts() if "tps_jac_orth" in cost_type]
    if len(tps_jac_orth_cost) > 0:
        tps_jac_orth_cost = np.sum(tps_jac_orth_cost)
        f_jacs = f.compute_jacobian(closing_pts)
        tps_jac_orth_err = []
        for jac in f_jacs:
            tps_jac_orth_err.extend((jac.dot(jac.T) - np.eye(3)).flatten())
        tps_jac_orth_err = np.asarray(tps_jac_orth_err)
        tps_jac_orth_cost2 = np.square( 10.0 * tps_jac_orth_err ).sum()

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])

    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps", tps_cost, tps_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", tps_rel_pts_costs, tps_rel_pts_costs2)
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", tps_jac_orth_cost, tps_jac_orth_cost2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps (matching)", np.abs(matching_err).min(), np.abs(matching_err).max())
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", np.abs(tps_rel_pts_err).min(), np.abs(tps_rel_pts_err).max())
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", np.abs(tps_jac_orth_err).min(), np.abs(tps_jac_orth_err).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())

    return traj, obj_value, tps_rel_pts_costs, tps_cost

def decomp_fit_tps_follow_finger_pts_trajs(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2demo_finger_pts_trajs, old_traj,
                                         f, closing_pts=None,
                                         no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                          alpha=settings.ALPHA, beta_pos=settings.BETA_POS, gamma=settings.GAMMA, plotting=False):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()

    n_steps = old_traj.shape[0]
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[1] == len(dof_inds)
    for flr2demo_finger_pts_traj in flr2demo_finger_pts_trajs:
        for demo_finger_pts_traj in flr2demo_finger_pts_traj.values():
            assert len(demo_finger_pts_traj)== n_steps
    assert len(flr2finger_link_names) == len(flr2demo_finger_pts_trajs)

    # expand these
    (n,d) = f.x_na.shape
    bend_coefs = np.ones(d) * f.bend_coef if np.isscalar(f.bend_coef) else f.bend_coef
    rot_coefs = np.ones(d) * f.rot_coef if np.isscalar(f.rot_coef) else f.rot_coef
    if f.wt_n is None:
        wt_n = np.ones(n)
    else:
        wt_n = f.wt_n
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))

    init_traj = old_traj.copy()
    N = f.N
    init_z = f.z

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    #ipy.embed();
    tps_request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "m_ext" : n,
            "n_ext" : d,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        {
            "type" : "tps",
            "name" : "tps",
            "params" : {"x_na" : [row.tolist() for row in f.x_na],
                        "y_ng" : [row.tolist() for row in f.y_ng],
                        "bend_coefs" : bend_coefs.tolist(),
                        "rot_coefs" : rot_coefs.tolist(),
                        "wt_n" : [row.tolist() for row in wt_n],
                        "N" : [row.tolist() for row in N],
                        "alpha" : alpha,
            }
        }
        ],
        "constraints" : [
        ],
    }

    traj_request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        ],
        "constraints" : [
        ],
    }
    if use_collision_cost:
        traj_request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })
    if joint_vel_limits is not None:
        tps_request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })
        traj_request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })
    if closing_pts is not None:
        tps_request["costs"].append(
            {
                "type":"tps_jac_orth",
                "params":  {
                            "tps_cost_name":"tps",
                            "pts":closing_pts.tolist(),
                            "coeffs":[10.0]*len(closing_pts),
                            }
            })

    # Now that we've made the initial request that is the same every iteration,
    # we make the loop and add on the things that change.

    # TODO - Set this traj_dim automatically.
    traj_dim = old_traj.size
    lambdas = np.zeros((traj_dim,))
    nu = 100.0
    traj_diff_thresh = 1e-3*traj_dim
    max_iter = 15
    tps_traj = init_traj
    traj_traj = init_traj

    for itr in range(max_iter):
      #if itr is 3:
      #  nu = 0.001
      traj_request_i = copy.deepcopy(traj_request)
      flr2transformed_finger_pts_traj = {}
      for finger_lr in 'lr':
        flr2transformed_finger_pts_traj[finger_lr] = f.transform_points(np.concatenate(flr2demo_finger_pts_trajs[0][finger_lr], axis=0)).reshape((-1,4,3))
      # TODO - Probs not the right thing to do...
      flr2transformed_finger_pts_trajs = [flr2transformed_finger_pts_traj]

      traj_request_i["init_info"] = {
            "type":"given_traj",
            "data":[x.tolist() for x in traj_traj],
        }

      for (flr2finger_link_name, flr2transformed_finger_pts_traj) in zip(flr2finger_link_names, flr2transformed_finger_pts_trajs):
          for finger_lr, finger_link_name in flr2finger_link_name.items():
              finger_rel_pts = flr2finger_rel_pts[finger_lr]
              transformed_finger_pts_traj = flr2transformed_finger_pts_traj[finger_lr]
              for (i_step, finger_pts) in enumerate(transformed_finger_pts_traj):
                  if start_fixed and i_step == 0:
                      continue
                  traj_request_i["costs"].append(
                      {"type":"rel_pts",
                       "params":{
                          "xyzs":finger_pts.tolist(),
                          "rel_xyzs":finger_rel_pts.tolist(),
                          "link":finger_link_name,
                          "timestep":i_step,
                          "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4,
                        }
                      })
      s_traj = json.dumps(traj_request_i);
      print 'Setting up and solving Traj SQP'
      with openravepy.RobotStateSaver(robot):
         with util.suppress_stdout():
          prob = trajoptpy.ConstructProblem(s_traj, robot.GetEnv())
          if plotting:
            viewer = trajoptpy.GetViewer(robot.GetEnv())
            trajoptpy.SetInteractive(True)
          result = trajoptpy.OptimizeTrajProblem(prob, (-lambdas).tolist())

      traj_traj = result.GetTraj()
      print traj_traj.shape
      tps_rel_pts_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "rel_pts"])
      collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]


      ########### PLOT TRAJ TRAJECTORY HERE ############

      # TODO - Double check if this should be column major ('C') or 'F'.
      traj_diff = tps_traj.flatten('C') - traj_traj.flatten('C')
      abs_traj_diff = sum(abs(traj_diff))
      print "Absolute difference between trajectories: ", abs_traj_diff
      #print "Traj diffs: ", traj_diff[-20:]
      print "Lambdas: ", lambdas[-20:]

      tps_request_i = copy.deepcopy(tps_request)
      tps_request_i["init_info"] = {
            "type":"given_traj",
            "data":[(x).tolist() for x in tps_traj],
            "data_ext":[(row).tolist() for row in f.z]
        }
      for (flr2finger_link_name, flr2demo_finger_pts_traj) in zip(flr2finger_link_names, flr2demo_finger_pts_trajs):
          for finger_lr, finger_link_name in flr2finger_link_name.items():
              finger_rel_pts = flr2finger_rel_pts[finger_lr]
              demo_finger_pts_traj = flr2demo_finger_pts_traj[finger_lr]
              for (i_step, demo_finger_pts) in enumerate(demo_finger_pts_traj):
                  if start_fixed and i_step == 0:
                      continue
                  tps_request_i["costs"].append(
                      {"type":"tps_rel_pts",
                       "params":{
                         "tps_cost_name":"tps",
                         "src_xyzs":demo_finger_pts.tolist(),
                         "rel_xyzs":finger_rel_pts.tolist(),
                         "link":finger_link_name,
                         "timestep":i_step,
                         "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4,
                        }
                      })
      s_tps = json.dumps(tps_request_i)
      print 'Setting up and solving TPS Problem'
      with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
          prob = trajoptpy.ConstructProblem(s_tps, robot.GetEnv())
          if plotting:
            viewer = trajoptpy.GetViewer(robot.GetEnv())
            trajoptpy.SetInteractive(True)
          result = trajoptpy.OptimizeTrajProblem(prob, lambdas.tolist())
      tps_traj = result.GetTraj()
      f.z = result.GetExt()
      theta = N.dot(f.z)
      f.trans_g = theta[0,:]
      f.lin_ag = theta[1:d+1,:]
      f.w_ng = theta[d+1:]
      ######### PLOT TPS TRAJ HERE ############

      traj_diff = tps_traj.flatten('C') - traj_traj.flatten('C')
      abs_traj_diff = sum(abs(traj_diff))
      print "Absolute difference between trajectories: ", abs_traj_diff
      #print "Traj diffs: ", traj_diff[-20:]
      print "Lambdas: ", lambdas[-20:]
      lambdas = lambdas - nu * traj_diff
      if abs_traj_diff < traj_diff_thresh:
        print "TRAJECTORIES CONVERGED"
        break

    print 'Done optimizing'
    traj = traj_traj

    tps_rel_pts_err = []
    with openravepy.RobotStateSaver(robot):
        for (flr2finger_link_name, flr2demo_finger_pts_traj) in zip(flr2finger_link_names, flr2demo_finger_pts_trajs):
            for finger_lr, finger_link_name in flr2finger_link_name.items():
                finger_link = robot.GetLink(finger_link_name)
                finger_rel_pts = flr2finger_rel_pts[finger_lr]
                demo_finger_pts_traj = flr2demo_finger_pts_traj[finger_lr]
                for (i_step, demo_finger_pts) in enumerate(demo_finger_pts_traj):
                    if start_fixed and i_step == 0:
                        continue
                    robot.SetDOFValues(traj[i_step], dof_inds)
                    new_hmat = finger_link.GetTransform()
                    tps_rel_pts_err.append(f.transform_points(demo_finger_pts) - (new_hmat[:3,3][None,:] + finger_rel_pts.dot(new_hmat[:3,:3].T)))
    tps_rel_pts_err = np.concatenate(tps_rel_pts_err, axis=0)
    tps_rel_pts_costs2 = (beta_pos/n_steps) * np.square(tps_rel_pts_err).sum() # TODO don't square n_steps

    print 'Getting TPS Cost'
    tps_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "tps"])
    tps_cost2 = alpha * f.get_objective().sum()
    matching_err = f.transform_points(f.x_na) - f.y_ng

    print 'Getting Joint Vel Cost'
    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/(n_steps-1)) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    tps_jac_orth_cost = [cost_val for (cost_type, cost_val) in result.GetCosts() if "tps_jac_orth" in cost_type]
    if len(tps_jac_orth_cost) > 0:
        tps_jac_orth_cost = np.sum(tps_jac_orth_cost)
        f_jacs = f.compute_jacobian(closing_pts)
        tps_jac_orth_err = []
        for jac in f_jacs:
            tps_jac_orth_err.extend((jac.dot(jac.T) - np.eye(3)).flatten())
        tps_jac_orth_err = np.asarray(tps_jac_orth_err)
        tps_jac_orth_cost2 = np.square( 10.0 * tps_jac_orth_err ).sum()

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])

    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps", tps_cost, tps_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", tps_rel_pts_costs, tps_rel_pts_costs2)
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", tps_jac_orth_cost, tps_jac_orth_cost2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps (matching)", np.abs(matching_err).min(), np.abs(matching_err).max())
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", np.abs(tps_rel_pts_err).min(), np.abs(tps_rel_pts_err).max())
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", np.abs(tps_jac_orth_err).min(), np.abs(tps_jac_orth_err).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())

    return traj, obj_value, tps_rel_pts_costs, tps_cost




def decomp_fit_tps_trajopt(robot, manip_name, flr2finger_link_names, flr2finger_rel_pts, flr2demo_finger_pts_trajs, old_traj,
                           f, closing_pts=None,
                           no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                           alpha=settings.ALPHA, beta_pos=settings.BETA_POS, gamma=settings.GAMMA, plotting=False):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()

    n_steps = old_traj.shape[0]
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[1] == len(dof_inds)
    for flr2demo_finger_pts_traj in flr2demo_finger_pts_trajs:
        for demo_finger_pts_traj in flr2demo_finger_pts_traj.values():
            assert len(demo_finger_pts_traj)== n_steps
    assert len(flr2finger_link_names) == len(flr2demo_finger_pts_trajs)

    # expand these
    (n,d) = f.x_na.shape
    bend_coefs = np.ones(d) * f.bend_coef if np.isscalar(f.bend_coef) else f.bend_coef
    rot_coefs = np.ones(d) * f.rot_coef if np.isscalar(f.rot_coef) else f.rot_coef
    if f.wt_n is None:
        wt_n = np.ones(n)
    else:
        wt_n = f.wt_n
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))

    init_traj = old_traj.copy()
    N = f.N
    init_z = f.z

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]
    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/(n_steps-1)]}
        },
        ],
        "constraints" : [
        ],
    }
    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })
    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    # Now that we've made the initial request that is the same every iteration,
    # we make the loop and add on the things that change.

    traj_dim = old_traj.size
    lambdas = np.zeros((traj_dim,))
    nu = 100.0
    traj_diff_thresh = 1e-3*traj_dim
    max_iter = 15
    tps_traj = init_traj
    traj_traj = init_traj

    for itr in range(max_iter):
      request_i = copy.deepcopy(request)
      flr2transformed_finger_pts_traj = {}
      for finger_lr in 'lr':
        flr2transformed_finger_pts_traj[finger_lr] = f.transform_points(np.concatenate(flr2demo_finger_pts_trajs[0][finger_lr], axis=0)).reshape((-1,4,3))
      # TODO - Probs not the right thing to do...
      flr2transformed_finger_pts_trajs = [flr2transformed_finger_pts_traj]

      request_i["init_info"] = {
            "type":"given_traj",
            "data":[x.tolist() for x in traj_traj],
        }

      for (flr2finger_link_name, flr2transformed_finger_pts_traj) in zip(flr2finger_link_names, flr2transformed_finger_pts_trajs):
          for finger_lr, finger_link_name in flr2finger_link_name.items():
              finger_rel_pts = flr2finger_rel_pts[finger_lr]
              transformed_finger_pts_traj = flr2transformed_finger_pts_traj[finger_lr]
              for (i_step, finger_pts) in enumerate(transformed_finger_pts_traj):
                  if start_fixed and i_step == 0:
                      continue
                  request_i["costs"].append(
                      {"type":"rel_pts",
                       "params":{
                          "xyzs":finger_pts.tolist(),
                          "rel_xyzs":finger_rel_pts.tolist(),
                          "link":finger_link_name,
                          "timestep":i_step,
                          "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4,
                        }
                      })
      s_traj = json.dumps(request_i);
      print 'Setting up and solving Traj SQP'
      with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
          prob = trajoptpy.ConstructProblem(s_traj, robot.GetEnv())
          if plotting:
            viewer = trajoptpy.GetViewer(robot.GetEnv())
            trajoptpy.SetInteractive(True)
          result = trajoptpy.OptimizeTrajProblem(prob, (-lambdas).tolist())

      traj_traj = result.GetTraj()
      print traj_traj.shape

      ########### PLOT TRAJ TRAJECTORY HERE ############

      # TODO - Double check if this should be column major ('C') or 'F'.
      traj_diff = tps_traj.flatten('C') - traj_traj.flatten('C')
      abs_traj_diff = sum(abs(traj_diff))
      print "Absolute difference between trajectories: ", abs_traj_diff
      #print "Traj diffs: ", traj_diff[-20:]
      print "Lambdas: ", lambdas[-20:]

      tps_traj = result.GetTraj()
      f.z = result.GetExt()
      theta = N.dot(f.z)
      f.trans_g = theta[0,:]
      f.lin_ag = theta[1:d+1,:]
      f.w_ng = theta[d+1:]
      ######### PLOT TPS TRAJ HERE ############

      traj_diff = tps_traj.flatten('C') - traj_traj.flatten('C')
      abs_traj_diff = sum(abs(traj_diff))
      print "Absolute difference between trajectories: ", abs_traj_diff
      #print "Traj diffs: ", traj_diff[-20:]
      print "Lambdas: ", lambdas[-20:]
      lambdas = lambdas - nu * traj_diff
      if abs_traj_diff < traj_diff_thresh:
        print "TRAJECTORIES CONVERGED"
        break

    print 'Done optimizing'

