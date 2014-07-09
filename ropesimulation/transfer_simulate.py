from __future__ import division

import sim_util
import h5py
import numpy as np
from constants import ROPE_RADIUS, DS_SIZE, JOINT_LENGTH_PER_STEP, FINGER_CLOSE_RATE
from rapprentice import ropesim, resampling, clouds
from rapprentice import tps_registration, planning
from rope_utils import get_closing_pts, get_closing_inds
from rapprentice.util import redprint, yellowprint
from numpy import asarray
from rapprentice import math_utils as mu

import IPython as ipy

class Transfer(object):
    def __init__(self, args_eval, action_solvers, register_tps):
        self.args_eval = args_eval
        self.action_solvers = action_solvers
        self.register_tps = register_tps

class TrajectoryResult(object):
    def __init__(self, success, feasible, misgrasp, full_trajs):
        self.success = success
        self.feasible = feasible
        self.misgrasp = misgrasp
        self.full_trajs = full_trajs
        
class BatchTransferSimulate(object):
    ### START interface
    def __init__(self, transfer, sim_env):
        self.transfer = transfer
        self.sim_env = sim_env
        self.actions = h5py.File(self.transfer.args_eval.actionfile, 'r')
        self.job_inputs = []
    
    def add_transfer_simulate_job(self, state, action, next_state_id):
        self.job_inputs.append((state, action, next_state_id))
    
    def get_results(self, animate=False):
        results = []
        while len(self.job_inputs) > 0:
            state, action, next_state_id = self.job_inputs.pop(0)
            success, feasible, misgrasp, full_trajs, next_state = self.compute_trans_traj(self.sim_env, state, action, self.transfer.args_eval, next_state_id, animate=animate)
            trajectory_result = TrajectoryResult(success, feasible, misgrasp, full_trajs)
            results.append((trajectory_result, next_state, next_state_id))
        return results
    ## END interface

    def get_action_cloud(self, sim_env, action, args_eval):
        rope_nodes = self.get_action_rope_nodes(sim_env, action, args_eval)
        cloud = ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample_rad=args_eval.upsample_rad)
        return cloud
    
    def get_action_cloud_ds(self, sim_env, action, args_eval):
        if args_eval.downsample:
            ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
            return self.actions[action]['inv'][ds_key]['cloud_xyz']
        else:
            return self.get_action_cloud(sim_env, action, args_eval)
    
    def get_action_rope_nodes(self, sim_env, action, args_eval):
        rope_nodes = self.actions[action]['cloud_xyz'][()]
        return ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample=args_eval.upsample)
    
    def compute_trans_traj(self, sim_env, state, action, args_eval, next_state_id, transferopt=None, animate=False, interactive=False, simulate=True, replay_full_trajs=None):
        alpha = args_eval.alpha
        beta_pos = args_eval.beta_pos
        beta_rot = args_eval.beta_rot
        gamma = args_eval.gamma
        if transferopt is None:
            transferopt = args_eval.transferopt
        
        seg_info = self.actions[action]
        if simulate:
            sim_util.reset_arms_to_side(sim_env)
        
        cloud_dim = 6 if args_eval.use_color else 3
        old_cloud = self.get_action_cloud_ds(sim_env, action, args_eval)[:,:cloud_dim]
        old_rope_nodes = self.get_action_rope_nodes(sim_env, action, args_eval)
        
        new_cloud = state.cloud
        new_cloud = new_cloud[:,:cloud_dim]
        
        sim_env.set_rope_state(state)
    
        handles = []
        if animate:
            # color code: r demo, y transformed, g transformed resampled, b new
            handles.append(sim_env.env.plot3(old_cloud[:,:3], 2, (1,0,0)))
            handles.append(sim_env.env.plot3(new_cloud[:,:3], 2, new_cloud[:,3:] if args_eval.use_color else (0,0,1)))
            sim_env.viewer.Step()
        
        closing_inds = get_closing_inds(seg_info)
        closing_hmats = {}
        for lr in closing_inds:
            if closing_inds[lr] != -1:
                closing_hmats[lr] = seg_info["%s_gripper_tool_frame"%lr]['hmat'][closing_inds[lr]]
        
        miniseg_intervals = []
        for lr in 'lr':
            miniseg_intervals.extend([(i_miniseg_lr, lr, i_start, i_end) for (i_miniseg_lr, (i_start, i_end)) in enumerate(zip(*sim_util.split_trajectory_by_lr_gripper(seg_info, lr)))])
        # sort by the start of the trajectory, then by the length (if both trajectories start at the same time, the shorter one should go first), and then break ties by executing the right trajectory first
        miniseg_intervals = sorted(miniseg_intervals, key=lambda (i_miniseg_lr, lr, i_start, i_end): (i_start, i_end-i_start, {'l':'r', 'r':'l'}[lr]))
        
        miniseg_interval_groups = []
        for (curr_miniseg_interval, next_miniseg_interval) in zip(miniseg_intervals[:-1], miniseg_intervals[1:]):
            curr_i_miniseg_lr, curr_lr, curr_i_start, curr_i_end = curr_miniseg_interval
            next_i_miniseg_lr, next_lr, next_i_start, next_i_end = next_miniseg_interval
            if len(miniseg_interval_groups) > 0 and curr_miniseg_interval in miniseg_interval_groups[-1]:
                continue
            curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%curr_lr][curr_i_end])
            next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%next_lr][next_i_end])
            miniseg_interval_group = [curr_miniseg_interval]
            if not curr_gripper_open and not next_gripper_open and curr_lr != next_lr and curr_i_start < next_i_end and next_i_start < curr_i_end:
                miniseg_interval_group.append(next_miniseg_interval)
            miniseg_interval_groups.append(miniseg_interval_group)

        f, corr = self.transfer.register_tps(sim_env, state, action, args_eval, reg_type='bij')
        
        success = True
        feasible = True
        misgrasp = False
        full_trajs = []
        obj_values = []
        for i_miniseg_group, miniseg_interval_group in enumerate(miniseg_interval_groups):
            if not simulate or replay_full_trajs is None: # we are not simulating, we still want to compute the costs
                group_full_trajs = []
                for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
                    ee_link_name = "%s_gripper_tool_frame"%lr
            
                    ################################    
                    redprint("Generating %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
                    
                    # figure out how we're gonna resample stuff
                    old_arm_traj = asarray(seg_info[manip_name][i_start - int(i_start > 0):i_end+1])
                    if not sim_util.arm_moved(old_arm_traj):
                        continue
                    old_finger_traj = sim_util.gripper_joint2gripper_l_finger_joint_values(seg_info['%s_gripper_joint'%lr][i_start - int(i_start > 0):i_end+1])[:,None]
                    _, timesteps_rs = sim_util.unif_resample(old_arm_traj, JOINT_LENGTH_PER_STEP)
                
                    ### Generate fullbody traj
                    old_arm_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_arm_traj)), old_arm_traj)
    
                    if animate:
                        handles.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 2, old_cloud[:,3:] if args_eval.use_color else (1,1,0)))
                        new_cloud_rs = corr.dot(new_cloud)
                        handles.append(sim_env.env.plot3(new_cloud_rs[:,:3], 2, new_cloud_rs[:,3:] if args_eval.use_color else (0,1,0)))
                        handles.extend(sim_util.draw_grid(sim_env, old_cloud[:,:3], f))
                    
                    x_na = old_cloud
                    y_ng = (corr/corr.sum(axis=1)[:,None]).dot(new_cloud)
                    bend_coef = f._bend_coef
                    rot_coef = f._rot_coef
                    wt_n = f._wt_n.copy()
                    
                    interest_pts_inds = np.zeros(len(old_cloud), dtype=bool)
                    if lr in closing_hmats:
                        interest_pts_inds += np.apply_along_axis(np.linalg.norm, 1, old_cloud - closing_hmats[lr][:3,3]) < 0.05
        
                    interest_pts_err_tol = 0.0025
                    max_iters = 5 if transferopt != "pose" else 0
                    penalty_factor = 10.0
                    
                    if np.any(interest_pts_inds):
                        for _ in range(max_iters):
                            interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                            if np.all(interest_pts_errs < interest_pts_err_tol):
                                break
                            redprint("TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                            wt_n[interest_pts_inds] *= penalty_factor
                            self.transfer.action_solvers[action].solve(wt_n, y_ng, bend_coef, rot_coef, f)
                            
            
                    old_ee_traj = asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'][i_start - int(i_start > 0):i_end+1])
                    transformed_ee_traj = f.transform_hmats(old_ee_traj)
                    transformed_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(transformed_ee_traj)), transformed_ee_traj))
                     
                    if animate:
                        handles.append(sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0)))
                        handles.append(sim_env.env.drawlinestrip(transformed_ee_traj[:,:3,3], 2, (1,1,0)))
                        handles.append(sim_env.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                        sim_env.viewer.Step()
                    
                    print "planning pose trajectory following"
                    dof_inds = sim_util.dof_inds_from_name(sim_env.robot, manip_name)
                    joint_ind = sim_env.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
                    init_arm_traj = old_arm_traj_rs.copy()
                    init_arm_traj[:,dof_inds.index(joint_ind)] = sim_env.robot.GetDOFLimits([joint_ind])[0][0]
                    new_arm_traj, obj_value, pose_errs = planning.plan_follow_traj(sim_env.robot, manip_name, sim_env.robot.GetLink(ee_link_name), transformed_ee_traj_rs, init_arm_traj, 
                                                                                   start_fixed=i_miniseg_lr!=0,
                                                                                   use_collision_cost=False,
                                                                                   beta_pos=beta_pos, beta_rot=beta_rot)
                    
                    if transferopt == 'finger' or transferopt == 'joint':
                        old_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(old_ee_traj)), old_ee_traj))
                        old_finger_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_finger_traj)), old_finger_traj)
                        flr2old_finger_pts_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_traj_rs, old_finger_traj_rs))
                        
                        flr2transformed_finger_pts_traj_rs = {}
                        flr2finger_link = {}
                        flr2finger_rel_pts = {}
                        for finger_lr in 'lr':
                            flr2transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            flr2finger_link[finger_lr] = sim_env.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,finger_lr))
                            flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)
                        
                        if animate:
                            handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2old_finger_pts_traj_rs, (1,0,0)))
                            handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                            sim_env.viewer.Step()
                            
                        # enable finger DOF and extend the trajectories to include the closing part only if the gripper closes at the end of this minisegment
                        next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                        if not sim_env.sim.is_grabbing_rope(lr) and not next_gripper_open:
                            manip_name = manip_name + "+" + "%s_gripper_l_finger_joint"%lr
                            
                            old_finger_closing_traj_start = old_finger_traj_rs[-1][0]
                            old_finger_closing_traj_target = sim_util.get_binary_gripper_angle(sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]))
                            old_finger_closing_traj_rs = np.linspace(old_finger_closing_traj_start, old_finger_closing_traj_target, np.ceil(abs(old_finger_closing_traj_target - old_finger_closing_traj_start) / FINGER_CLOSE_RATE))[:,None]
                            closing_n_steps = len(old_finger_closing_traj_rs)
                            old_ee_closing_traj_rs = np.tile(old_ee_traj_rs[-1], (closing_n_steps,1,1))
                            flr2old_finger_pts_closing_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_closing_traj_rs, old_finger_closing_traj_rs))
                              
                            init_traj = np.r_[np.c_[new_arm_traj,                                   old_finger_traj_rs],
                                                np.c_[np.tile(new_arm_traj[-1], (closing_n_steps,1)), old_finger_closing_traj_rs]]
                            # init_traj = np.r_[np.c_[init_arm_traj,                                   old_finger_traj_rs],
                            #                     np.c_[np.tile(init_arm_traj[-1], (closing_n_steps,1)), old_finger_closing_traj_rs]]
                            flr2transformed_finger_pts_closing_traj_rs = {}
                            for finger_lr in 'lr':
                                flr2old_finger_pts_traj_rs[finger_lr] = np.r_[flr2old_finger_pts_traj_rs[finger_lr], flr2old_finger_pts_closing_traj_rs[finger_lr]]
                                flr2transformed_finger_pts_closing_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_closing_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                                flr2transformed_finger_pts_traj_rs[finger_lr] = np.r_[flr2transformed_finger_pts_traj_rs[finger_lr],
                                                                                      flr2transformed_finger_pts_closing_traj_rs[finger_lr]]
                            
                            if animate:
                                handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2old_finger_pts_closing_traj_rs, (1,0,0)))
                                handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_closing_traj_rs, (0,1,0)))
                                sim_env.viewer.Step()
                        else:
                            init_traj = new_arm_traj
                            # init_traj = init_arm_traj
                        
                        new_traj, obj_value, pose_errs = planning.plan_follow_finger_pts_traj(sim_env.robot, manip_name, 
                                                                                              flr2finger_link, flr2finger_rel_pts, 
                                                                                              flr2transformed_finger_pts_traj_rs, init_traj, 
                                                                                              use_collision_cost=False,
                                                                                              start_fixed=i_miniseg_lr!=0,
                                                                                              beta_pos=beta_pos, gamma=gamma)
    
                        
                        if transferopt == 'joint':
                            print "planning joint TPS and finger points trajectory following"
                            new_traj, f, new_N_z, \
                            obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                               x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=None,
                                                                                                               start_fixed=i_miniseg_lr!=0,
                                                                                                               alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                            if np.any(interest_pts_inds):
                                for _ in range(max_iters):
                                    interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                                    if np.all(interest_pts_errs < interest_pts_err_tol):
                                        break
                                    redprint("Joint TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                                    wt_n[interest_pts_inds] *= penalty_factor
                                    new_traj, f, new_N_z, \
                                    obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                                       x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=new_N_z,
                                                                                                                       start_fixed=i_miniseg_lr!=0,
                                                                                                                       alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                        # else:
                        #     obj_value += alpha * planning.tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n)
                        
                        if animate:
                            flr2new_transformed_finger_pts_traj_rs = {}
                            for finger_lr in 'lr':
                                flr2new_transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2new_transformed_finger_pts_traj_rs, (0,1,1)))
                            sim_env.viewer.Step()
                    else:
                        new_traj = new_arm_traj
                    
                    obj_values.append(obj_value)
                    
                    f._bend_coef = bend_coef
                    f._rot_coef = rot_coef
                    f._wt_n = wt_n
                    
                    full_traj = (new_traj, sim_util.dof_inds_from_name(sim_env.robot, manip_name))
                    group_full_trajs.append(full_traj)
        
                    if animate:
                        handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, full_traj)[:,:3,3], 2, (0,0,1)))
                        flr2new_finger_pts_traj = sim_util.get_finger_pts_traj(sim_env, lr, full_traj)
                        handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2new_finger_pts_traj, (0,0,1)))
                        sim_env.viewer.Step()
                full_traj = sim_util.merge_full_trajs(group_full_trajs)
            else:
                full_traj = replay_full_trajs[i_miniseg_group]
            full_trajs.append(full_traj)
            
            if not simulate:
                if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                    return np.inf
                else:
                    continue
    
            for (i_miniseg_lr, lr, _, _) in miniseg_interval_group:
                redprint("Executing %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
            
            if len(full_traj[0]) > 0:
                # if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                #     redprint("Trajectory not feasible")
                #     feasible = False
                #     success = False
                # else:  # Only execute feasible trajectories
                first_miniseg = True
                for (i_miniseg_lr, _, _, _) in miniseg_interval_group:
                    first_miniseg &= i_miniseg_lr == 0
                if len(full_traj[0]) > 0:
                    success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive, max_cart_vel_trans_traj=.05 if first_miniseg else .02)
    
            if not success: break
            
            for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
                next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end])
                if not sim_util.set_gripper_maybesim(sim_env, lr, next_gripper_open, curr_gripper_open, animate=animate):
                    redprint("Grab %s failed" % lr)
                    misgrasp = True
                    success = False
    
            if not success: break
    
        if not simulate:
            return np.sum(obj_values)
    
        sim_env.sim.settle(animate=animate)
        sim_env.sim.release_rope('l')
        sim_env.sim.release_rope('r')
        sim_util.reset_arms_to_side(sim_env)
        if animate:
            sim_env.viewer.Step()
        
        return success, feasible, misgrasp, full_trajs, self.get_state(sim_env, args_eval, next_state_id)
    
    def get_state(self, sim_env, args_eval, state_id):
        if args_eval.raycast:
            new_cloud, endpoint_inds = sim_env.sim.raycast_cloud(endpoints=3)
            if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
                return None
        else:
            new_cloud = sim_env.sim.observe_cloud(upsample=args_eval.upsample, upsample_rad=args_eval.upsample_rad)
            endpoint_inds = np.zeros(len(new_cloud), dtype=bool) # for now, args_eval.raycast=False is not compatible with args_eval.use_color=True
        if args_eval.use_color:
            new_cloud = color_cloud(new_cloud, endpoint_inds)
        new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE) if args_eval.downsample else new_cloud
        new_rope_nodes = sim_env.sim.rope.GetControlPoints()
        new_rope_nodes= ropesim.observe_cloud(new_rope_nodes, sim_env.sim.rope_params.radius, upsample=args_eval.upsample)
        init_rope_nodes = sim_env.sim.rope_pts
        rope_params = args_eval.rope_params
        tfs = sim_util.get_rope_transforms(sim_env)
        state = sim_util.RopeState("eval_%i"%state_id, new_cloud_ds, new_rope_nodes, init_rope_nodes, rope_params, tfs)
        return state
