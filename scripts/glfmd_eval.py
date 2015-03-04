#!/usr/bin/env python

from __future__ import division

from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.environment.robot_world import RealRobotWorld
from lfd.demonstration.demonstration import Demonstration, SceneState, AugmentedTrajectory
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import PoseTrajectoryTransferer
import scipy.spatial.distance as ssd
import sys
import colorsys
import pickle
import datetime
from lfd.lfmd.analyze_data import dof_val_cost, align_trajs
from lfd.registration import tps_experimental, hc
from lfd.rapprentice import plotting_openrave, clouds
import IPython as ipy

import argparse

from lfd.rapprentice import berkeley_pr2
try:
    from lfd.rapprentice import pr2_trajectories, PR2
    import rospy
except:
    print "Couldn't import ros stuff"

import cloudprocpy, trajoptpy, openravepy
import os, numpy as np, h5py, time
from numpy import asarray
import importlib

def parse_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("method", type=str, choices=['rpm', 'pair', 'cov'], help="tps-rpm, pairwise l2, or pairwise l2 with covariance")
    parser.add_argument('actionfiles', type=str, nargs='*')

    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--execution", type=int, default=0)

    parser.add_argument("--cloud_proc_func", default="extract_red")
    parser.add_argument("--cloud_proc_mod", default="lfd.rapprentice.cloud_proc_funcs")
    parser.add_argument("--downsample_size", type=float, default=0.025)
    
    parser.add_argument("--fake_data_segment",type=str, default=None)
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")

    parser.add_argument("--beta_pos", type=float, default=1000000.0)
    parser.add_argument("--beta_rot", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--use_collision_cost", type=int, default=1)

    parser.add_argument("--pos_coef", type=list, default=[0,0,1], help="coefficient for dtw position cost")
    parser.add_argument("--rot_coef", type=float, default=0, help="coefficient for dtw rotation cost")
    parser.add_argument("--pos_vel_coef", type=float, default=1, help="coefficient for dtw position velocity cost")
    parser.add_argument("--rot_vel_coef", type=float, default=0, help="coefficient for dtw rotation velocity cost")
    parser.add_argument("--downsample_traj", type=int, default=1, help="downsample demonstration trajectory by this factor")

    args = parser.parse_args()
    return args

def register_scenes(sim, reg_factory, scene_state):
    print "registering all scenes... "
    regs = []
    demos = []
    
    for action, demo in reg_factory.demos.iteritems():
        reg = reg_factory.register(demo, scene_state)
        regs.append(reg)
        demos.append(demo)
    q_values, regs, demos = zip(*sorted([(reg.f.get_objective().sum(), reg, demo) for (reg, demo) in zip(regs, demos)]))
    print "done"
    
    return regs, demos

def setup_demos(args, robot, actionfile=None):
    if actionfile is None:
        actionfile = args.actionfiles[0]
    actions = h5py.File(actionfile)
    
    demos = {}
    for action, seg_info in actions.iteritems():
#         if 'towel' in action and '00_seg' in action: continue
#         if 'towel' in action and '10_seg' in action: continue
#         if 'towel' in action and '20_seg' in action: continue
        if 'towel' in action and '30_seg' in action: continue
        # TODO
#         if 'seg00' not in action or 'failure' in action: continue
#         if len(demos) > 5: break
        full_cloud = seg_info['cloud_xyz'][()]
#         #TODO
#         from lfd.rapprentice import clouds
#         cloud = np.r_[clouds.downsample(full_cloud[full_cloud[:,1] < 0], 0.015),
#                       clouds.downsample(full_cloud[full_cloud[:,1] > 0], 0.05)]
#         scene_state = SceneState(cloud)
#         scene_state = SceneState(full_cloud, downsample_size=args.downsample_size)
#         if 'towel' in action:
#             scene_state = create_scene_state(full_cloud, args.downsample_size)
#         else:
        scene_state = SceneState(full_cloud, downsample_size=args.downsample_size)
        scene_state.rgb = seg_info['rgb'][()] #TODO
        lr2arm_traj = {}
        lr2finger_traj = {}
        lr2ee_traj = {}
        lr2open_finger_traj = {}
        lr2close_finger_traj = {}
        for lr in 'lr':
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            lr2arm_traj[lr] = np.asarray(seg_info[arm_name])
            lr2finger_traj[lr] = sim_util.gripper_joint2gripper_l_finger_joint_values(np.asarray(seg_info['%s_gripper_joint'%lr]))[:,None]
            lr2ee_traj[lr] = np.asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'])
            lr2open_finger_traj[lr] = np.zeros(len(lr2finger_traj[lr]), dtype=bool)
            lr2close_finger_traj[lr] = np.zeros(len(lr2finger_traj[lr]), dtype=bool)
            opening_inds, closing_inds = sim_util.get_opening_closing_inds(lr2finger_traj[lr])
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
#             # opening_inds/closing_inds are indices before the opening/closing happens, so increment those indices (if they are not out of bound)
#             opening_inds = np.clip(opening_inds+1, 0, len(lr2finger_traj[lr])-1) # TODO figure out if +1 is necessary
#             closing_inds = np.clip(closing_inds+1, 0, len(lr2finger_traj[lr])-1)
        aug_traj = AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        demo = Demonstration(action, scene_state, aug_traj)
        demos[action] = demo

    return demos

def setup_lfd_environment_sim(args):
    actions = h5py.File(args.actionfiles[0], 'r')
    
    if args.fake_data_segment is None:
        fake_data_segment = actions.keys()[0]
    else:
        fake_data_segment = args.fake_data_segment
    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, fake_data_segment, args.fake_data_transform) 
    table_height = init_rope_xyz[:,2].min() #TODO table height

    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
#     sim_objs.append(BoxSimulationObject("table", [1, 0, table_height -.1], [.85, .85, .1], dynamic=False))
    
    sim = DynamicSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    lfd_env = environment.LfdEnvironment(sim, world, downsample_size=args.downsample_size)
    
    _, seg_info = actions.items()[0]
    init_joint_names = np.asarray(seg_info["joint_states"]["name"])
    init_joint_values = np.asarray(seg_info["joint_states"]["position"][0])
    dof_inds = sim_util.dof_inds_from_name(sim.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    sim.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
    sim_util.reset_arms_to_side(sim)
    actions.close()
    
    if args.animation:
        sim.create_viewer()
    return lfd_env, sim

def align_aug_trajs(aug_trajs, active_lr, pos_coef, rot_coef, pos_vel_coef, rot_vel_coef, downsample_traj):
    coefs = [pos_coef, rot_coef, pos_vel_coef, rot_vel_coef, 0, 0]
    if downsample_traj > 1:
        ds_aug_trajs = [aug_traj.get_resampled_traj(np.arange(aug_traj.n_steps)[::downsample_traj]) for aug_traj in aug_trajs]
    else:
        ds_aug_trajs = aug_trajs
    if len(active_lr) == 1:
        dof_cost = lambda aug_traj1, aug_traj2, t1, t2: dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[0], coefs)
    elif len(active_lr) == 2:
        dof_cost = lambda aug_traj1, aug_traj2, t1, t2: dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[0], coefs) + dof_val_cost(aug_traj1, aug_traj2, t1, t2, active_lr[1], coefs)
    else:
        raise RuntimeError
    ds_trajs_timesteps_rs = align_trajs(ds_aug_trajs, dof_cost)
    
    aligned_aug_trajs = []
    for aug_traj, ds_traj_timesteps_rs in zip(aug_trajs, ds_trajs_timesteps_rs):
        traj_timesteps_rs0 = np.arange(downsample_traj*(len(ds_traj_timesteps_rs)-1)+1)
        ds_traj_timesteps_rs, ds_traj_timesteps_rs0 = np.unique(ds_traj_timesteps_rs, return_index=True)
        ds_traj_timesteps_rs0 = ds_traj_timesteps_rs0.astype(float)
        ds_traj_timesteps_rs0[:-1] += (np.diff(ds_traj_timesteps_rs0)-1)/2
        traj_timesteps_rs = np.interp(traj_timesteps_rs0, ds_traj_timesteps_rs0*downsample_traj, ds_traj_timesteps_rs*downsample_traj)
        aligned_aug_trajs.append(aug_traj.get_resampled_traj(traj_timesteps_rs))
    return aligned_aug_trajs

def l2_callback(f, y_md, sim):
    handles = []
    x_ld = f.x_la
    xwarped_nd = f.transform_points()
    handles.append(sim.env.plot3(x_ld, 5, (1,0,0)))
    handles.append(sim.env.plot3(xwarped_nd, 5, (0,1,0)))
    handles.append(sim.env.plot3(y_md, 5, (0,0,1)))
    handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_ld.min(axis=0) - .1, x_ld.max(axis=0) + .1, xres=.1, yres=.1, zres=.1))
    sim.viewer.Step()
    return handles

def multi_l2_callback(f_k, y_md, p_ktd, sim, colors):
    handles = []
    for f, p_td, color in zip(f_k, p_ktd, colors):
        x_ld = f.x_la
        xwarped_nd = f.transform_points()
#         handles.append(sim.env.plot3(x_ld, 3, color))
        handles.append(sim.env.plot3(xwarped_nd, 5, color))
        pwarped_td = f.transform_points(p_td)
        handles.append(sim.env.drawlinestrip(pwarped_td, 2, color))
#         handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_ld.min(axis=0) - .1, x_ld.max(axis=0) + .1, xres=.05, yres=.05, zres=.04, color=color))
    if y_md is not None:
        handles.append(sim.env.plot3(y_md, 8, (0,0,1)))
    sim.viewer.Step()
    return handles

def show_image(rgb, name):
    import cv2
    cv2.imshow(name, rgb)
    cv2.waitKey()

def create_scene_state(new_xyz, downsample_size):
    cloud = new_xyz
    med = np.median(cloud[:,0])
    cloud = cloud[np.logical_and((med - .2) < cloud[:,0], cloud[:,0] < (med + .2)), :]
    tape_cloud = np.r_[cloud[cloud[:,0] < (cloud[:,0].min() + 0.04)], cloud[cloud[:,0] > (cloud[:,0].max() - 0.04)]]
    cloud = np.r_[clouds.downsample(new_xyz, downsample_size), clouds.downsample(tape_cloud, 0.015)]
    scene_state = SceneState(cloud)
    return scene_state

def main():
    args = parse_input_args()
    
    trajoptpy.SetInteractive(args.interactive)
    
    lfd_env, sim = setup_lfd_environment_sim(args)

    traj_transferer = PoseTrajectoryTransferer(sim)

    if args.execution:
        rospy.init_node("exec_task",disable_signals=True)
        pr2 = PR2.PR2()
        robot = pr2.robot
        
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()
    
        world = RealRobotWorld(pr2)
        lfd_env_real = environment.LfdEnvironment(world, sim, downsample_size=args.downsample_size)
    
    for actionfile in args.actionfiles:
        action_name = os.path.splitext(os.path.basename(actionfile))[0]

        sim_util.reset_arms_to_side(sim)
        if args.execution:
            pr2.head.set_pan_tilt(0,1.05)
            pr2.rarm.goto_posture('side')
            pr2.larm.goto_posture('side')
            pr2.rgrip.set_angle(0.54800022)
            pr2.lgrip.set_angle(0.54800022)
            pr2.join_all()
            time.sleep(.5)
            pr2.update_rave()

        demos_dict = setup_demos(args, sim.robot, actionfile=actionfile)
        
        # sorted by demo name
        _, demos, aug_trajs = zip(*sorted([(name, demo, demo.aug_traj) for name, demo in demos_dict.items()], key=lambda x: x[0]))
    
        if 'towelone_0.h5' in actionfile:
            active_lr = 'lr'
        elif 'towelthree_0.h5' in actionfile:
            active_lr = 'l'
        else:
            active_lr = 'r'
        
        if args.execution:        
            rgb, depth = grabber.getRGBD()
            T_w_k = berkeley_pr2.get_kinect_transform(robot)
            cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
            cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)
            new_xyz = cloud_proc_func(rgb, depth, T_w_k)
            cloud_dict = {}
            cloud_dict['rgb'] = rgb
            cloud_dict['depth'] = depth
            cloud_dict['T_w_k'] = T_w_k
            cloud_dict['new_xyz'] = new_xyz
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            pickle.dump(cloud_dict, open("clouds/" + action_name + "_" + args.method + "_" + st + ".pkl", "wb" ))
            test_scene_state = SceneState(new_xyz, downsample_size=args.downsample_size)
        else:
#             clouds = h5py.File('../bigdata/misc/ropeclutter_0.h5', 'r')
#             test_scene_state = SceneState(clouds['ropeclutter_00_seg00']['cloud_xyz'][()], downsample_size=args.downsample_size)
            if 'towel' in action_name:
                cloud_dict = pickle.load(open("clouds/toweltwomarker_0_cov_2015-03-03_20:03:40.pkl", "rb" ))
                cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
                cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)
                rgb = cloud_dict['rgb']
                depth = cloud_dict['depth']
                T_w_k = cloud_dict['T_w_k']
                new_xyz = cloud_proc_func(rgb, depth, T_w_k)
                
                # move tape
                cloud = new_xyz
                med = np.median(cloud[:,0])
                cloud = cloud[np.logical_and((med - .2) < cloud[:,0], cloud[:,0] < (med + .2)), :]
                tape_inds = np.logical_or(cloud[:,0] < (cloud[:,0].min() + 0.06), cloud[:,0] > (cloud[:,0].max() - 0.06))
                towel_cloud = cloud[~tape_inds, :]
                tape_cloud = cloud[tape_inds, :]
                tape_cloud[:,1] -= .2
                new_xyz = np.r_[tape_cloud, towel_cloud]
    #             if 'towel' in action_name:
    #                 test_scene_state = create_scene_state(new_xyz, args.downsample_size)
    #             else:
                test_scene_state = SceneState(new_xyz, downsample_size=args.downsample_size)
            else:
                test_scene_state = demos[-1].scene_state
        handles = []
        
        if args.method == 'rpm':
            regs, demos = register_scenes(sim, TpsRpmRegistrationFactory(demos_dict), test_scene_state)
            reg = regs[0]
            demo = demos[0]
            def tps_callback(i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad, sim):
                registration_plot_cb(sim, x_nd, y_md, f)
            reg_factory = TpsRpmRegistrationFactory(demos_dict, reg_init=100, reg_final=10, rad_init=0.1, rad_final=0.001) #, em_iter=5, reg_init=0.1, reg_final=0.0001, rad_init=0.01, rad_final=0.00005)
            reg = reg_factory.register(demo, test_scene_state, callback=tps_callback, args=(sim,))
            
            test_aug_traj = traj_transferer.transfer(reg, demo, plotting=True)
        else:
            sys.stdout.write("aligning trajectories... ")
            sys.stdout.flush()
            aligned_aug_trajs = align_aug_trajs(aug_trajs, active_lr, np.asarray(args.pos_coef), args.rot_coef, args.pos_vel_coef, args.rot_vel_coef, args.downsample_traj)
            sys.stdout.flush()
            print "done"
    
            demo_colors = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, len(demos), endpoint=False)]
            
    #         for demo in demos:
    #             show_image(demo.scene_state.rgb, demo.name)
    #         show_image(test_scene_state.rgb, "test")
    
    #         for demo, aug_traj, color in zip(demos, aug_trajs, demo_colors):
    #             handles.append(sim.env.plot3(demo.scene_state.cloud, 5, color))
    #             for lr in active_lr:
    #                 handles.append(sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 2, color))
    #             lfd_env.execute_augmented_trajectory(aug_traj, step_viewer=args.animation, interactive=args.interactive)
    #         sim.viewer.Step()
    
    #         for demo, aug_traj, color in zip(demos, aligned_aug_trajs, demo_colors):
    #             handles.append(sim.env.plot3(demo.scene_state.cloud, 5, color))
    #             for lr in active_lr:
    #                 handles.append(sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 5, color))
    #         sim.viewer.Idle()
    
            x_kld = []
            for demo in demos:
                x_kld.append(demo.scene_state.cloud)
            y_md = test_scene_state.cloud
            
            assert len(active_lr) == 1
            p_ktd = []
            for aligned_aug_traj in aligned_aug_trajs:
                p_ktd.append(aligned_aug_traj.lr2ee_traj[active_lr][:,:3,3])
            p_ktd = np.array(p_ktd)
            
            traj_rad = 0.2
            w_kt = []
            for x_ld, p_td in zip(x_kld, p_ktd):
                dist_tl = ssd.cdist(p_td, x_ld, 'sqeuclidean')
                dist_t = np.min(dist_tl, axis=1)
                w_kt.append(np.exp(-dist_t / (traj_rad**2)))
            w_kt = np.asarray(w_kt)
            w_t = w_kt.mean(axis=0)
            
            f_k = []
            for x_ld in x_kld:
                from lfd.rapprentice import clouds
                f = tps_experimental.ThinPlateSpline(x_ld, clouds.downsample(x_ld, .05))
    #             f = tps_experimental.ThinPlateSpline(x_ld, x_ld)
                f_k.append(f)
            if args.method == 'pair':
                f_k = tps_experimental.pairwise_tps_l2_cov(x_kld, y_md, p_ktd, f_init_k=f_k, n_iter=2, cov_coef=0, w_t=None, reg_init=100, reg_final=10, rad_init=1, rad_final=.1, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                                           callback=l2_callback, args=(sim,), multi_callback=multi_l2_callback, multi_args=(sim, demo_colors))
            else:
                if 'towel' in action_name:
                    cov_coef = .3
                else:
                    cov_coef = .1
                f_k = tps_experimental.pairwise_tps_l2_cov(x_kld, y_md, p_ktd, f_init_k=f_k, n_iter=2, cov_coef=cov_coef, w_t=w_t, reg_init=100, reg_final=10, rad_init=1, rad_final=.1, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                                           callback=l2_callback, args=(sim,), multi_callback=multi_l2_callback, multi_args=(sim, demo_colors))
            
            handles.extend(multi_l2_callback(f_k, y_md, p_ktd, sim, demo_colors))
    
    #         f_k = tps_experimental.pairwise_tps_l2_cov(x_kld, y_md, p_ktd, f_init_k=f_k, n_iter=10, cov_coef=.0001, reg_init=1, reg_final=.1, rad_init=.1, rad_final=.01, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
    #                                                    callback=l2_callback, args=(sim,), multi_callback=multi_l2_callback, multi_args=(sim, demo_colors))
            
    #         f_k = []
    #         for (x_ld, p_td, ctrl_nd) in zip(x_kld, p_ktd, x_kld):
    #             n, d = ctrl_nd.shape
    #             f = tps_experimental.tps_l2(x_ld, y_md, ctrl_nd=ctrl_nd, n_iter=10, opt_iter=100, reg_init=0.1, reg_final=0.01, rad_init=.1, rad_final=.01, rot_reg=np.r_[1e-4, 1e-4, 1e-1], callback=l2_callback, args=(sim,))
    #             f_k.append(f)
    #         f_k = hc.groupwise_tps_hc2(x_kld, y_md=y_md, f_init_k=f_k, opt_iter=100, reg=0.01, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
    #                                        callback=multi_l2_callback, args=(p_ktd, sim, demo_colors))
    #         
    #         f_k = hc.groupwise_tps_hc2_cov(x_kld, p_ktd, f_init_k=f_k, y_md=y_md, opt_iter=100, reg=0.01, rot_reg=np.r_[1e-4, 1e-4, 1e-1], cov_coef=.0001, 
    #                                        callback=multi_l2_callback, args=(p_ktd, sim, demo_colors), multi_callback=multi_l2_callback, multi_args=(sim, demo_colors))
    # #         multi_l2_callback(f_k, y_md, p_ktd, sim, demo_colors)
            
            # hack to compute test_aug_traj
            class IdentityRegistration(object):
                def __init__(self):
                    self.f = tps_experimental.ThinPlateSpline(np.zeros((4,3)), np.zeros((4,3)))
            reg = IdentityRegistration()
            test_aug_traj = aligned_aug_trajs[-1]
            not_active_lr = 'r' if active_lr is 'l' else 'l'
            del test_aug_traj.lr2arm_traj[not_active_lr]
            del test_aug_traj.lr2ee_traj[not_active_lr]
            fp_ktd = []
            for f, p_td in zip(f_k, p_ktd):
                fp_ktd.append(f.transform_points(p_td))
            fp_ktd = np.asarray(fp_ktd)
            handles.append(sim.env.drawlinestrip(np.mean(fp_ktd, axis=0), 5, (0,0,1)))
            test_aug_traj.lr2ee_traj[active_lr][:,:3,3] = np.mean(fp_ktd, axis=0)
            test_demo = demos[-1]
            test_demo.aug_traj = test_aug_traj
            test_aug_traj = traj_transferer.transfer(reg, test_demo)
        
        lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
        
        # sim.viewer.Idle()
        
        if args.execution:
            lfd_env_real.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
    
        # ipy.embed()
        
        handles[:] = []

    if args.execution:
        pr2.head.set_pan_tilt(0,1.05)
        pr2.rarm.goto_posture('side')
        pr2.larm.goto_posture('side')
        pr2.rgrip.set_angle(0.54800022)
        pr2.lgrip.set_angle(0.54800022)
        pr2.join_all()
        time.sleep(.5)
        pr2.update_rave()

if __name__ == '__main__':
    main()
