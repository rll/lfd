from __future__ import division

import sim_util
import h5py
import numpy as np
from constants import ROPE_RADIUS, DS_SIZE, JOINT_LENGTH_PER_STEP, FINGER_CLOSE_RATE
from rapprentice import ropesim, resampling, clouds
from rapprentice import tps_registration, registration, planning
from rope_utils import get_closing_pts, get_closing_inds
from rapprentice.util import redprint, yellowprint
from numpy import asarray
from rapprentice import math_utils as mu
from rapprentice.registration import fit_ThinPlateSpline
import sys, os

from IPython import parallel
from IPython.parallel import interactive

import IPython as ipy

class TransferSimulateResult(object):
    def __init__(self, success, feasible, misgrasp, full_trajs, state):
        self.success = success
        self.feasible = feasible
        self.misgrasp = misgrasp
        self.full_trajs = full_trajs
        self.state = state

class TransferSimulate(object):
    def __init__(self, transfer, sim_env):
        self.transfer = transfer
        self.sim_env = sim_env
    
    def transfer_simulate(self, state, action, next_state_id, animate=False, interactive=False, simulate=True, use_collision_cost=False, replay_full_trajs=None):
        args_eval = self.transfer.args_eval
        alpha = args_eval.alpha
        beta_pos = args_eval.beta_pos
        beta_rot = args_eval.beta_rot
        gamma = args_eval.gamma
        transferopt = args_eval.transferopt
        
        seg_info = self.transfer.actions[action]
        if simulate:
            sim_util.reset_arms_to_side(self.sim_env)
        
        cloud_dim = 6 if args_eval.use_color else 3
        old_cloud = self.transfer.get_action_cloud_ds(action, args_eval)[:,:cloud_dim]
        old_rope_nodes = self.transfer.get_action_rope_nodes(action, args_eval)
        
        new_cloud = state.cloud
        new_cloud = new_cloud[:,:cloud_dim]
        
        self.sim_env.set_rope_state(state.rope_state)
    
        handles = []
        if animate:
            # color code: r demo, y transformed, g transformed resampled, b new
            handles.append(self.sim_env.env.plot3(old_cloud[:,:3], 2, (1,0,0)))
            handles.append(self.sim_env.env.plot3(new_cloud[:,:3], 2, new_cloud[:,3:] if args_eval.use_color else (0,0,1)))
            self.sim_env.viewer.Step()
        
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

        f, corr = self.transfer.register_tps(state, action, args_eval, reg_type='bij')
        
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
                        handles.append(self.sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 2, old_cloud[:,3:] if args_eval.use_color else (1,1,0)))
                        new_cloud_rs = corr.dot(new_cloud)
                        handles.append(self.sim_env.env.plot3(new_cloud_rs[:,:3], 2, new_cloud_rs[:,3:] if args_eval.use_color else (0,1,0)))
                        handles.extend(sim_util.draw_grid(self.sim_env, old_cloud[:,:3], f))
                    
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
                            fit_ThinPlateSpline(x_na, y_ng, bend_coef=bend_coef, rot_coef=rot_coef, wt_n=wt_n)
                            
            
                    old_ee_traj = asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'][i_start - int(i_start > 0):i_end+1])
                    transformed_ee_traj = f.transform_hmats(old_ee_traj)
                    transformed_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(transformed_ee_traj)), transformed_ee_traj))
                     
                    if animate:
                        handles.append(self.sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0)))
                        handles.append(self.sim_env.env.drawlinestrip(transformed_ee_traj[:,:3,3], 2, (1,1,0)))
                        handles.append(self.sim_env.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                        self.sim_env.viewer.Step()
                    
                    print "planning pose trajectory following"
                    dof_inds = sim_util.dof_inds_from_name(self.sim_env.robot, manip_name)
                    joint_ind = self.sim_env.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
                    init_arm_traj = old_arm_traj_rs.copy()
                    init_arm_traj[:,dof_inds.index(joint_ind)] = self.sim_env.robot.GetDOFLimits([joint_ind])[0][0]
                    new_arm_traj, obj_value, pose_errs = planning.plan_follow_traj(self.sim_env.robot, manip_name, self.sim_env.robot.GetLink(ee_link_name), transformed_ee_traj_rs, init_arm_traj, 
                                                                                   start_fixed=i_miniseg_lr!=0,
                                                                                   use_collision_cost=use_collision_cost,
                                                                                   beta_pos=beta_pos, beta_rot=beta_rot)
                    
                    if transferopt == 'finger' or transferopt == 'joint':
                        old_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(old_ee_traj)), old_ee_traj))
                        old_finger_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_finger_traj)), old_finger_traj)
                        flr2old_finger_pts_traj_rs = sim_util.get_finger_pts_traj(self.sim_env, lr, (old_ee_traj_rs, old_finger_traj_rs))
                        
                        flr2transformed_finger_pts_traj_rs = {}
                        flr2finger_link = {}
                        flr2finger_rel_pts = {}
                        for finger_lr in 'lr':
                            flr2transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            flr2finger_link[finger_lr] = self.sim_env.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,finger_lr))
                            flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)
                        
                        if animate:
                            handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2old_finger_pts_traj_rs, (1,0,0)))
                            handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                            self.sim_env.viewer.Step()
                            
                        # enable finger DOF and extend the trajectories to include the closing part only if the gripper closes at the end of this minisegment
                        next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                        if not self.sim_env.sim.is_grabbing_rope(lr) and not next_gripper_open:
                            manip_name = manip_name + "+" + "%s_gripper_l_finger_joint"%lr
                            
                            old_finger_closing_traj_start = old_finger_traj_rs[-1][0]
                            old_finger_closing_traj_target = sim_util.get_binary_gripper_angle(sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]))
                            old_finger_closing_traj_rs = np.linspace(old_finger_closing_traj_start, old_finger_closing_traj_target, np.ceil(abs(old_finger_closing_traj_target - old_finger_closing_traj_start) / FINGER_CLOSE_RATE))[:,None]
                            closing_n_steps = len(old_finger_closing_traj_rs)
                            old_ee_closing_traj_rs = np.tile(old_ee_traj_rs[-1], (closing_n_steps,1,1))
                            flr2old_finger_pts_closing_traj_rs = sim_util.get_finger_pts_traj(self.sim_env, lr, (old_ee_closing_traj_rs, old_finger_closing_traj_rs))
                              
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
                                handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2old_finger_pts_closing_traj_rs, (1,0,0)))
                                handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2transformed_finger_pts_closing_traj_rs, (0,1,0)))
                                self.sim_env.viewer.Step()
                        else:
                            init_traj = new_arm_traj
                            # init_traj = init_arm_traj
                        
                        print "planning finger trajectory following"
                        new_traj, obj_value, pose_errs = planning.plan_follow_finger_pts_traj(self.sim_env.robot, manip_name, 
                                                                                              flr2finger_link, flr2finger_rel_pts, 
                                                                                              flr2transformed_finger_pts_traj_rs, init_traj, 
                                                                                              use_collision_cost=use_collision_cost,
                                                                                              start_fixed=i_miniseg_lr!=0,
                                                                                              beta_pos=beta_pos, gamma=gamma)
    
                        
                        if transferopt == 'joint':
                            print "planning joint TPS and finger points trajectory following"
                            new_traj, f, new_N_z, \
                            obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(self.sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                               x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=None,
                                                                                                               use_collision_cost=use_collision_cost,
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
                                    obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(self.sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                                       x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=new_N_z,
                                                                                                                       use_collision_cost=use_collision_cost,
                                                                                                                       start_fixed=i_miniseg_lr!=0,
                                                                                                                       alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                        # else:
                        #     obj_value += alpha * planning.tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n)
                        
                        if animate:
                            flr2new_transformed_finger_pts_traj_rs = {}
                            for finger_lr in 'lr':
                                flr2new_transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2new_transformed_finger_pts_traj_rs, (0,1,1)))
                            self.sim_env.viewer.Step()
                    else:
                        new_traj = new_arm_traj
                    
                    obj_values.append(obj_value)
                    
                    f._bend_coef = bend_coef
                    f._rot_coef = rot_coef
                    f._wt_n = wt_n
                    
                    full_traj = (new_traj, sim_util.dof_inds_from_name(self.sim_env.robot, manip_name))
                    group_full_trajs.append(full_traj)
        
                    if animate:
                        handles.append(self.sim_env.env.drawlinestrip(sim_util.get_ee_traj(self.sim_env, lr, full_traj)[:,:3,3], 2, (0,0,1)))
                        flr2new_finger_pts_traj = sim_util.get_finger_pts_traj(self.sim_env, lr, full_traj)
                        handles.extend(sim_util.draw_finger_pts_traj(self.sim_env, flr2new_finger_pts_traj, (0,0,1)))
                        self.sim_env.viewer.Step()
                full_traj = sim_util.merge_full_trajs(group_full_trajs)
            else:
                full_traj = replay_full_trajs[i_miniseg_group]
            full_trajs.append(full_traj)
            
            if not simulate:
                if not eval_util.traj_is_safe(self.sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                    return np.inf
                else:
                    continue
    
            for (i_miniseg_lr, lr, _, _) in miniseg_interval_group:
                redprint("Executing %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
            
            if len(full_traj[0]) > 0:
                # if not eval_util.traj_is_safe(self.sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                #     redprint("Trajectory not feasible")
                #     feasible = False
                #     success = False
                # else:  # Only execute feasible trajectories
                first_miniseg = True
                for (i_miniseg_lr, _, _, _) in miniseg_interval_group:
                    first_miniseg &= i_miniseg_lr == 0
                if len(full_traj[0]) > 0:
                    success &= sim_util.sim_full_traj_maybesim(self.sim_env, full_traj, animate=animate, interactive=interactive, max_cart_vel_trans_traj=.05 if first_miniseg else .02)
    
            if not success: break
            
            for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
                next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end])
                if not sim_util.set_gripper_maybesim(self.sim_env, lr, next_gripper_open, curr_gripper_open, animate=animate):
                    redprint("Grab %s failed" % lr)
                    misgrasp = True
                    success = False
    
            if not success: break
    
        if not simulate:
            return np.sum(obj_values)
    
        self.sim_env.sim.settle(animate=animate)
        self.sim_env.sim.release_rope('l')
        self.sim_env.sim.release_rope('r')
        sim_util.reset_arms_to_side(self.sim_env)
        if animate:
            self.sim_env.viewer.Step()
        
        return TransferSimulateResult(success, feasible, misgrasp, full_trajs, self.sim_env.observe_scene(id=next_state_id, **vars(args_eval)))

class BatchTransferSimulate(object):
    def __init__(self, transfer, sim_env, max_queue_size = 100, profile='ssh'):
        self.transfer = transfer
        self.sim_env = sim_env
        self.max_queue_size = max_queue_size

        # create clients and views
        self.rc = parallel.Client(profile=profile)
        self.dv = self.rc[:]
        self.v = self.rc.load_balanced_view()
 
        # add module paths to the engine paths
        modules = ['lfd', 'reinforcement-lfd']
        module_paths = []
        for module in modules:
            paths = [path for path in sys.path if module == os.path.split(path)[1]]
            assert len(paths) > 0
            module_paths.append(paths[0]) # add the first module path only
 
        @interactive
        def engine_add_module_paths(module_paths):
            import sys
            sys.path.extend(module_paths)
        self.dv.map_sync(engine_add_module_paths, [module_paths]*len(self.dv))

        @interactive
        def engine_initialize(id, transfer, sim_env):
            from ropesimulation.transfer_simulate import TransferSimulate
            global transfer_simulate
            transfer_simulate = TransferSimulate(transfer, sim_env)
            transfer_simulate.transfer.initialize()
            transfer_simulate.sim_env.initialize()
        transfer.args_eval.actionfile = os.path.abspath(transfer.args_eval.actionfile)
        self.dv.map_sync(engine_initialize, self.rc.ids, [transfer]*len(self.dv), [sim_env]*len(self.dv))
        
        self.pending = set()
        
    def queue_transfer_simulate(self, state, action, next_state_id): # TODO optional arguments
        self.wait_while_queue_is_full()
        
        @interactive
        def engine_transfer_simulate(state, action, next_state_id):
            global transfer_simulate
            return transfer_simulate.transfer_simulate(state, action, next_state_id)
        
        amr = self.v.map(engine_transfer_simulate, *[[e] for e in [state, action, next_state_id]])
        self.pending.update(amr.msg_ids)

    def wait_while_queue_size_above_size(self, queue_size):
        pending = self.pending.copy()
        while len(pending) > queue_size:
            try:
                self.rc.wait(pending, 1e-3)
            except parallel.TimeoutError:
                # ignore timeouterrors, since they only mean that at least one isn't done
                pass
            # finished is the set of msg_ids that are complete
            finished = pending.difference(self.rc.outstanding)
            # update pending to exclude those that just finished
            pending = pending.difference(finished)

    def wait_while_queue_is_full(self):
        self.wait_while_queue_size_above_size(self.max_queue_size)

    def wait_while_queue_is_nonempty(self):
        self.wait_while_queue_size_above_size(0)

    def get_results(self):
        results = []
        try:
            self.rc.wait(self.pending, 1e-3)
        except parallel.TimeoutError:
            # ignore timeouterrors, since they only mean that at least one isn't done
            pass
        # finished is the set of msg_ids that are complete
        finished = self.pending.difference(self.rc.outstanding)
        # update pending to exclude those that just finished
        self.pending = self.pending.difference(finished)
        for msg_id in finished:
            # we know these are done, so don't worry about blocking
            ar = self.rc.get_result(msg_id)
            results.extend(ar.result)
        return results
