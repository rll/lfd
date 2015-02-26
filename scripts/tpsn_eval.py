#!/usr/bin/env python

from __future__ import division

import numpy as np

from lfd.environment.simulation import DynamicSimulationRobotWorld
from lfd.environment.simulation_object import XmlSimulationObject
from lfd.environment import environment
from lfd.environment import sim_util
from lfd.environment.robot_world import RealRobotWorld
from lfd.demonstration.demonstration import Demonstration, SceneState, AugmentedTrajectory
from lfd.registration.registration import TpsRpmRegistrationFactory, TpsnRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.transfer.transfer import FingerTrajectoryTransferer
from lfd.transfer.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer
import pickle
import scipy.spatial.distance as ssd
import IPython as ipy

import argparse

# from lfd.rapprentice import registration, berkeley_pr2, \
#      animate_traj, ros2rave, plotting_openrave, task_execution, \
#      tps, func_utils, resampling, clouds
# from lfd.rapprentice import math_utils as mu
# from lfd.rapprentice.yes_or_no import yes_or_no
 
from lfd.rapprentice import berkeley_pr2
from lfd.registration import tps_experimental
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
    
    parser.add_argument('actionfile', type=str)
    parser.add_argument("spline_type", type=str, choices=['tps', 'tpsn'], default='tps', help="spline type for rpm")

    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--execution", type=int, default=0)

    parser.add_argument("--downsample_size", type=float, default=0.025)
    parser.add_argument("--normals_downsample_size", type=float, default=0.04)
    
    parser.add_argument("--fake_data_segment",type=str, default=None)
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")

    parser.add_argument("--beta_pos", type=float, default=1000000.0)
    parser.add_argument("--beta_rot", type=float, default=100.0)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--use_collision_cost", type=int, default=1)

    args = parser.parse_args()
    return args

def setup_demos(args, robot):
    actions = h5py.File(args.actionfile)
    
    demos = {}
    for action, seg_info in actions.iteritems():
        if 'towel' in args.actionfile and 'fold' not in action: continue
        #TODO: use foldfirst08
        full_cloud = seg_info['cloud_xyz'][()]
        scene_state = SceneState(full_cloud, downsample_size=args.downsample_size)
        # too slow
#         rgb = actions['demo1-seg02']['rgb'][()]
#         depth = actions['demo1-seg02']['depth'][()]
#         T_w_k = actions['demo1-seg02']['T_w_k'][()]
#         scene_state = create_scene_state(rgb, depth, T_w_k, args.downsample_size)
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
    actions = h5py.File(args.actionfile, 'r')
    
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

def register_scenes(sim, reg_factory, scene_state):
    print "registering all scenes... "
    regs = []
    demos = []
    
#     from lfd.rapprentice import plotting_openrave
#     def plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, demo):
#         handles = []
#         handles.append(sim.env.plot3(x_nd, 5, (1,0,0,1)))
#         handles.append(sim.env.plot3(f.transform_points(x_nd), 5, (0,1,0,1)))
#         handles.append(sim.env.plot3(y_md, 5, (0,0,1,1)))
#         handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_nd.min(axis=0) - .1, x_nd.max(axis=0) + .1, xres = .05, yres = .05, zres = .04))
#         aug_traj = demo.aug_traj
#         for lr in 'lr':
#             handles.append(sim.env.drawlinestrip(aug_traj.lr2ee_traj[lr][:,:3,3], 2, (1,0,0)))
#             handles.append(sim.env.drawlinestrip(f.transform_hmats(aug_traj.lr2ee_traj[lr])[:,:3,3], 2, (0,1,0)))
#         print "plot_cb"
#         sim.viewer.Idle()
    
    for action, demo in reg_factory.demos.iteritems():
#         reg = reg_factory.register(demo, scene_state, plotting=5, plot_cb=lambda x_nd, y_md, targ_Nd, corr_nm, wt_n, f: plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, demo))
        reg = reg_factory.register(demo, scene_state)
        regs.append(reg)
        demos.append(demo)
    q_values, regs, demos = zip(*sorted([(reg.f.get_objective().sum(), reg, demo) for (reg, demo) in zip(regs, demos)]))
    print "done"
    
    return regs, demos

# def parse_input_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("h5file", type=str)
#     parser.add_argument("--cloud_proc_func", default="extract_red")
#     parser.add_argument("--cloud_proc_mod", default="rapprentice.cloud_proc_funcs")
#         
#     parser.add_argument("--execution", type=int, default=0)
#     parser.add_argument("--animation", type=int, default=0)
#     parser.add_argument("--parallel", type=int, default=1)
#     
#     parser.add_argument("--prompt", action="store_true")
#     parser.add_argument("--show_neighbors", action="store_true")
#     parser.add_argument("--select_manual", action="store_true")
#     parser.add_argument("--log", action="store_true")
#     
#     parser.add_argument("--fake_data_segment",type=str)
#     parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
#         default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
#     
#     parser.add_argument("--interactive",action="store_true")
#     
#     args = parser.parse_args()
#     return args

from lfd.rapprentice import clouds
import cv2
def extract_red_obj_and_green_table(rgb, depth, T_w_k, plot=True):
    """
    extract red points and downsample
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    green_mask = (h > 30) & (h < 130) & (s > 50) & (h < 170) & (v < 240)
    
    valid_mask = depth > 0
    
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    z = xyz_w[:,:,2]   
    z0 = xyz_k[:,:,2]
#     if plot:
#         cv2.imshow("z0",z0/z0.max())
#         cv2.imshow("z",z/z.max())
#         cv2.imshow("rgb", rgb)
#         cv2.waitKey()
    
    height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter
    
    # bounding box mask in pixel space
    pix_bb_mins = [30, 30]
    pix_bb_maxs = [-30, -30]
    pix_bb_mask = np.ones_like(valid_mask, dtype=bool)
    pix_bb_mask[:pix_bb_mins[0], :] = False
    pix_bb_mask[pix_bb_maxs[0]:, :] = False
    pix_bb_mask[:, :pix_bb_mins[1]] = False
    pix_bb_mask[:, pix_bb_maxs[1]:] = False
    
    obj_mask = red_mask & height_mask & valid_mask & pix_bb_mask
#     if plot:
#         cv2.imshow("red",red_mask.astype('uint8')*255)
#         cv2.imshow("above_table", height_mask.astype('uint8')*255)
#         cv2.imshow("red and above table", obj_mask.astype('uint8')*255)
#         print "press enter to continue"
#         cv2.waitKey()

    obj_xyz = xyz_w[obj_mask]

#     padding = 0.10
#     bb_mins = np.min(obj_xyz.reshape((-1,3)), axis=0) - padding
#     bb_maxs = np.max(obj_xyz.reshape((-1,3)), axis=0) + padding
#     bb_mask = (xyz_w[:,:,0] > bb_mins[0]) & (xyz_w[:,:,0] < bb_maxs[0]) & \
#         (xyz_w[:,:,1] > bb_mins[1]) & (xyz_w[:,:,1] < bb_maxs[1])
#     table_mask = green_mask & bb_mask & valid_mask
    
#     # specialized table filter for curved table
#     green_max = np.max(xyz_w[green_mask & valid_mask], axis=0)
#     top_mask = xyz_w[:,:,2] > green_max[2] - 0.04
#     top_table_mask = green_mask & top_mask & valid_mask
#     top_table = xyz_w[top_table_mask]
#     top_table_center = (np.max(top_table.reshape((-1,3)), axis=0) + np.min(top_table.reshape((-1,3)), axis=0)) / 2
#     padding = 0.04
#     table_length = 0.9144
#     bb_mins = top_table_center - table_length/2 + padding
#     bb_maxs = top_table_center + table_length/2 - padding
#     bb_mask = (xyz_w[:,:,0] > bb_mins[0]) & (xyz_w[:,:,0] < bb_maxs[0]) & \
#         (xyz_w[:,:,1] > bb_mins[1]) & (xyz_w[:,:,1] < bb_maxs[1])
#     table_mask = green_mask & bb_mask & valid_mask
    table_mask = green_mask & valid_mask
    
    if plot:
        cv2.imshow("rgb", rgb)
#         cv2.imshow("top_table_mask", top_table_mask.astype('uint8')*255)
        cv2.imshow("obj_mask and rgb", obj_mask.astype('uint8')[:,:,None] * rgb)
        cv2.imshow("table_mask and rgb", table_mask.astype('uint8')[:,:,None] * rgb)
        print "press enter to continue"
        cv2.waitKey()
    
    table_xyz = xyz_w[table_mask]

    return obj_xyz, table_xyz

def compute_table_normals(table_xyz, object_xyz=None, relevant_radius=0.05, normals_downsample_size=0.04, window_radius=0.05):
    # if object_xyz is specified, it only consider normals whose locations are within relevant_radius from the object points
    z_sd = clouds.downsample(table_xyz, normals_downsample_size)
    if object_xyz is not None:
        dists = ssd.cdist(z_sd, object_xyz,'euclidean')
        z_sd = z_sd[np.any(dists < relevant_radius, axis=1),:]

    v_sd = []
    dists = ssd.cdist(z_sd, table_xyz,'euclidean')
    for dist in dists:
        nearby_pts = table_xyz[np.where(dist < window_radius)]
        centered_nearby_pts = nearby_pts - np.mean(nearby_pts, axis=0)
        _, _, v = np.linalg.svd(centered_nearby_pts.T.dot(centered_nearby_pts))
        v_sd.append(v[2,:].T)
    v_sd = np.asarray(v_sd)
    v_sd *= np.sign(v_sd[:,2])[:,None]
    return v_sd, z_sd

def create_scene_state(rgb, depth, T_w_k, downsample_size, normals_downsample_size):
    obj_xyz, table_xyz = extract_red_obj_and_green_table(rgb, depth, T_w_k, False)
#     obj_xyz = clouds.downsample(obj_xyz, downsample_size) #already done by next line
    scene_state = SceneState(obj_xyz, downsample_size=downsample_size)
    scene_state.normals, scene_state.sites = compute_table_normals(table_xyz, obj_xyz, normals_downsample_size=normals_downsample_size)
    return scene_state

def plot_cloud(sim, x_ld, u_rd, z_rd, color=(1,0,0,1), arrow_length=0.05, arrow_width=0.001):
    handles = []
    handles.append(sim.env.plot3(x_ld, 5, color))
#     handles.append(sim.env.plot3(z_sd, 5, (0,1,0,1)))
    for z_d, u_d in zip(z_rd, u_rd):
        handles.append(sim.env.drawarrow(z_d, (z_d + u_d*arrow_length), arrow_width, color))
    return handles

def plot_scene_state(sim, scene_state, color=(1,0,0,1), arrow_length=0.05, arrow_width=0.001):
    return plot_cloud(sim, scene_state.cloud, scene_state.normals, scene_state.sites, color=color, arrow_length=arrow_length, arrow_width=arrow_width)

def tps_callback(i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad, sim):
    registration_plot_cb(sim, x_nd, y_md, f)

def tpsn_callback(f, corr_lm, corr_rs, y_md, v_sd, z_sd, xtarg_ld, utarg_rd, wt_l, wt_r, reg, rad, radn, nu, i, i_em, sim):
    x_ld = f.x_la
    z_rd = f.z_ra
    u_rd = f.u_ra
    xwarped_ld = f.transform_points()
    uwarped_rd = f.transform_vectors()
    zwarped_rd = f.transform_points(z_rd)
    
    handles = []
    handles.extend(plot_cloud(sim, x_ld, u_rd, z_rd, (1,0,0)))
    handles.extend(plot_cloud(sim, y_md, v_sd, z_sd, (0,0,1)))
    handles.extend(plot_cloud(sim, xwarped_ld, uwarped_rd, zwarped_rd, (0,1,0)))
    
    sim.viewer.Step()

def main():
    args = parse_input_args()
    
    trajoptpy.SetInteractive(args.interactive)
    
#     ### START DEBUG
#     lfd_env, sim = setup_lfd_environment_sim(args)
#     demos = setup_demos(args, sim.robot)
#     reg_factory = TpsRpmRegistrationFactory(demos)
#     traj_transferer = FingerTrajectoryTransferer(sim)
#     
#     cloud_dict = pickle.load(open("clouds/curved_front_seg02.pkl", "rb" ))
#     rgb = cloud_dict['rgb']
#     depth = cloud_dict['depth']
#     T_w_k = cloud_dict['T_w_k']
#     test_scene_state = create_scene_state(rgb, depth, T_w_k, args.downsample_size)
#     
#     actions = h5py.File(args.actionfile, 'r')
#     rgb = actions['demo1-seg02']['rgb'][()]
#     depth = actions['demo1-seg02']['depth'][()]
#     T_w_k = actions['demo1-seg02']['T_w_k'][()]
#     obj_xyz, table_xyz = extract_red_obj_and_green_table(rgb, depth, T_w_k, False)
#     obj_xyz = clouds.downsample(obj_xyz, args.downsample_size)
#     x_ld = obj_xyz
#     u_rd, z_rd = compute_table_normals(table_xyz, obj_xyz)
# 
#     handles = []
#     handles.extend(plot_cloud(sim, x_ld, u_rd, z_rd, color=(1,0,0)))
#     handles.extend(plot_cloud(sim, y_md, v_sd, z_sd, color=(0,0,1)))
#     sim.viewer.Idle()
# 
# 
#     
#     reg_factory = TpsnRpmRegistrationFactory()
#     
#     # TODO: put this somehwere else
#     reg_factory.register(demo, test_scene_state, callback=callback)
# 
#     
#     f, corr_lm, corr_rs = tps_experimental.tpsn_rpm(x_ld, u_rd, z_rd, y_md, v_sd, z_sd, callback=tpsn_callback, args=(sim,))
# 
#     import IPython as ipy
#     ipy.embed()
#     
#     regs, demos = register_scenes(sim, reg_factory, test_scene_state)
#     demo = demos[28]
#     
# #     reg = reg_factory.register(demo, test_scene_state, callback=plot_cb)
#     reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
#     test_aug_traj = reg_and_traj_transferer.transfer(demo, test_scene_state, callback=plot_cb, plotting=True)
#     
#     lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
#     
#     import IPython as ipy
#     ipy.embed()
#     
#     import sys
#     sys.exit()
#     ### END DEBUG
    
    lfd_env, sim = setup_lfd_environment_sim(args)
    demos = setup_demos(args, sim.robot)
    if args.spline_type == 'tps':
        reg_factory = TpsRpmRegistrationFactory(demos)
        callback = tps_callback
    elif args.spline_type == 'tpsn':
        reg_factory = TpsnRpmRegistrationFactory(demos)
        callback = tpsn_callback
    else:
        raise NotImplementedError
    traj_transferer = FingerTrajectoryTransferer(sim)

    if args.execution:
        rospy.init_node("exec_task",disable_signals=True)
        pr2 = PR2.PR2()
        env = pr2.env
        robot = pr2.robot
        
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()
    
        world = RealRobotWorld(pr2)
        lfd_env_real = environment.LfdEnvironment(world, sim, downsample_size=args.downsample_size)
    
    while True:
        sim_util.reset_arms_to_side(sim)
        if args.execution:
            pr2.head.set_pan_tilt(0,1.2)
            pr2.rarm.goto_posture('side')
            pr2.larm.goto_posture('side')
            pr2.rgrip.set_angle(0.54800022)
            pr2.lgrip.set_angle(0.54800022)
            pr2.join_all()
            time.sleep(.5)
            pr2.update_rave()

        if args.execution:        
            rgb, depth = grabber.getRGBD()
            T_w_k = berkeley_pr2.get_kinect_transform(robot)
            test_scene_state = create_scene_state(rgb, depth, T_w_k, args.downsample_size, args.normals_downsample_size)
        else:
            cloud_dict = pickle.load(open("clouds/towel_curved_side_seg00.pkl", "rb" ))
            rgb = cloud_dict['rgb']
            depth = cloud_dict['depth']
            T_w_k = cloud_dict['T_w_k']
            test_scene_state = create_scene_state(rgb, depth, T_w_k, args.downsample_size, args.normals_downsample_size)
        
        handles = []
        handles.extend(plot_scene_state(sim, test_scene_state, color=(0,0,1)))
        sim.viewer.Step()
#         sim.viewer.Idle()
        
        # TODO: using TPS-RPM for action selection
        regs, demos = register_scenes(sim, TpsRpmRegistrationFactory(demos), test_scene_state)
        demo = demos[0]
        actions = h5py.File(args.actionfile, 'r')
        rgb = actions[demo.name]['rgb'][()]
        depth = actions[demo.name]['depth'][()]
        T_w_k = actions[demo.name]['T_w_k'][()]
        demo.scene_state = create_scene_state(rgb, depth, T_w_k, args.downsample_size, args.normals_downsample_size)

        handles.extend(plot_scene_state(sim, demo.scene_state, color=(1,0,0)))
        sim.viewer.Step()
#         sim.viewer.Idle()
        
#         reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
#         test_aug_traj = reg_and_traj_transferer.transfer(demo, test_scene_state, callback=callback, args=(sim,), plotting=True)
        reg = reg_factory.register(demo, test_scene_state, callback=callback, args=(sim,))
        if args.spline_type == 'tps':
            xwarped_ld = reg.f.transform_points(demo.scene_state.cloud)
            uwarped_rd = np.asarray([reg.f.compute_numerical_jacobian(z_d).dot(u_d) for z_d, u_d in zip(demo.scene_state.sites, demo.scene_state.normals)])
            zwarped_rd = reg.f.transform_points(demo.scene_state.sites)
        elif args.spline_type == 'tpsn':
            xwarped_ld = reg.f.transform_points()
            uwarped_rd = reg.f.transform_vectors()
            zwarped_rd = reg.f.transform_points(demo.scene_state.sites)
        else:
            raise NotImplementedError    
        handles.extend(plot_cloud(sim, xwarped_ld, uwarped_rd, zwarped_rd, (0,1,0)))
        sim.viewer.Idle()
        test_aug_traj = traj_transferer.transfer(reg, demo, plotting=True)
        
        lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
        
        sim.viewer.Idle()
        
        if args.execution:
            lfd_env_real.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)
    
        ipy.embed()
        
        if not args.execution:
            break

if __name__ == '__main__':
    main()
