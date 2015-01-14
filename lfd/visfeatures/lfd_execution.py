#!/usr/bin/env python

import argparse
usage="""
Actually run on the robot without pausing or animating 
./do_task.py ~/Data/overhand2/all.h5 --execution=1 --animation=0
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("h5file", type=str)
parser.add_argument("net_info_folder", type=str)  # folder that contains alexnet_deploy.prototxt, alexnet_train_iter_10000, mean.npy

parser.add_argument("--cloud_proc_func", default="extract_nongreen")
parser.add_argument("--cloud_proc_mod", default="towel_color_filters")
    
parser.add_argument("--animation", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--interactive",action="store_true")

parser.add_argument("--prompt", action="store_true")
parser.add_argument("--select_manual", action="store_true")


parser.add_argument("--use_vis", action="store_true")
parser.add_argument("--downsample", action="store_true")
parser.add_argument("--extra_corners", type=int, default=1)
parser.add_argument("--soft_limits", action="store_true")
args = parser.parse_args()

###################
import cloudprocpy, trajoptpy
import sys, os, numpy as np, h5py, time, IPython as ipy, math
import importlib

from rapprentice import registration, berkeley_pr2, task_execution
from lfd.rapprentice.util import redprint
from lfd.demonstration.demonstration import Demonstration, VisFeaturesSceneState
from lfd.environment import sim_util, exec_util
from lfd.registration.registration import TpsRpmRegistrationFactory

# Import ROS stuff
from rapprentice import PR2
import rospy

from alexnet_helper import get_alexnet
import plotting_plt_shhuang, vis_utils, misc_util

import subprocess

DS_SIZE = .025
POSTFIX = 'saved_alexnet_labels.h5' 
NUM_DEMOS_TO_CHECK = 5  # Change back to 3
HIST_COEFF = 10
TESTCLOUDS_F = 'saved_test_clouds_demsel.h5'
THRESHOLD = 0.15
BETA = 10

cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)

def find_closest_manual(demofile):
    "for now, just prompt the user"
    seg_names = demofile.keys()
    print "choose from the following options (type an integer)"
    for (i, seg_name) in enumerate(seg_names):
        print "%i: %s"%(i,seg_name)
    choice_ind = task_execution.request_int_in_range(len(seg_names))
    chosen_seg = seg_names[choice_ind] 
    return (chosen_seg, choice_ind)

def registration_cost(demo_info, test_state, tps_rpm_factory=None):
    # demo_info: a Demonstration with VisFeatureSceneState
    # test_state: a VisFeatureSceneState
    if tps_rpm_factory == None:
        tps_rpm_factory = Globals.tps_rpm_factory

    assert not args.use_vis or (demo_info.scene_state.alexnet_features is not None or test_state.alexnet_features is not None), "If using visual features, must provide SceneStates with precomputed Alexnet features"

    return tps_rpm_factory.viscost(demo_info, test_state)  # Computes objective cost, multiplying it by -1 because we're selecting for maximum score later

def get_demo_state(demofile, cloud_id, demos_saved_alexnet, net):
    # Returns a Demonstration object with VisFeaturesSceneState (alexnet_features = None if use_vis == False)
    if cloud_id not in Globals.demo_clouds:
        if demos_saved_alexnet != None:  # Clouds already have RGB, not BGR
            seg = demos_saved_alexnet[cloud_id]
            cloud_xyz = seg['cloud_xyz_full'][()]

            cloud_xyz_ds = seg['cloud_xyz_ds'][()]
            cloud_xyz_ds_100corners = None
            if args.extra_corners:
                cloud_xyz_ds_100corners = seg['cloud_xyz_ds_100corners'][()]

            alexnet_features = None
            alexnet_features_100corners = None
            if args.use_vis:
                features = {}
                features['fc7'] = seg['fc7_ds'][()]
                features['fc8'] = seg['fc8_ds'][()]
                alexnet_features = [seg['predicts_ds'][()], seg['scores_ds'][()], features, seg['validmask_ds'][()]]

                if args.extra_corners:
                    features_100corners = {}
                    features_100corners['fc7'] = seg['fc7_ds_100corners'][()]
                    features_100corners['fc8'] = seg['fc8_ds_100corners'][()]
                    alexnet_features_100corners = [seg['predicts_ds_100corners'][()], seg['scores_ds_100corners'][()], \
                                                   features_100corners, seg['validmask_ds_100corners'][()]]
        else:
            seg = demofile[cloud_id]
            redprint("Calculating Alexnet for demo")
            cloud_xyz = np.asarray(seg['cloud_xyz'][()])
            (cloud_xyz_ds, cloud_xyz_ds_100corners, alexnet_features, alexnet_features_100corners) = vis_util.get_alexnet_features(cloud_xyz, seg['rgb'][()], seg['T_w_k'][()], net, DS_SIZE, args)

        visfeatures_scene = VisFeaturesSceneState(cloud_xyz_ds, cloud_xyz_ds_100corners, cloud_xyz, alexnet_features,
                                                  alexnet_features_100corners)
        Globals.demo_clouds[cloud_id] = Demonstration(cloud_id, visfeatures_scene, None)

    return Globals.demo_clouds[cloud_id]

def find_closest_auto(test_state, demofile, demos_saved_alexnet, net, tps_rpm_factory):
    # Filter out all but 3 demos based on histogram matching
    demos_with_states = set(Globals.demo_clouds.keys())
    for k in demofile:  # Store all demos in Globals.demo_clouds
        if k not in demos_with_states:
            get_demo_state(demofile, k, demos_saved_alexnet, net)

    demo_keys = list(Globals.demo_clouds.keys())
    print demo_keys
    hist_distances = [(vis_utils.get_labels_distance(test_state.alexnet_features_wocorners[0], \
                                                     Globals.demo_clouds[k].scene_state.alexnet_features_wocorners[0]), i) \
                      for (i,k) in enumerate(demo_keys)]
    hist_distances = [(d,i) for (d,i) in hist_distances if not math.isnan(d)]
    print "Histogram distances:", hist_distances
    top_hist_distances = sorted(hist_distances)[:NUM_DEMOS_TO_CHECK]
    demo_keys_with_vals = [(demo_keys[i], x) for (x,i) in top_hist_distances]
    demo_keys = [demo_keys[i] for (x,i) in top_hist_distances]
    print "REMAINING DEMOS:", demo_keys_with_vals

    tps_rpm_factory.demos = Globals.demo_clouds
    tps_rpm_factory.n_iter = 10
    Globals.tps_rpm_factory = tps_rpm_factory

    if args.parallel:
        from joblib import Parallel, delayed
        costs = Parallel(n_jobs=5,verbose=100)(delayed(registration_cost)(Globals.demo_clouds[k], test_state) for k in demo_keys)
    else:
        print "Not running in parallel"
        costs = [registration_cost(Globals.demo_clouds[k], test_state, tps_rpm_factory) for k in demo_keys]
   
    print demo_keys
    print "old costs for selection:", costs
    costs = [(c + HIST_COEFF * top_hist_distances[i][0], i) for (i,c) in enumerate(costs) if not math.isnan(c)]
    print "costs for selection", costs
    sorted_costs = sorted(costs, key=lambda x: x[0])   
    return [(demo_keys[i], i) for (c,i) in sorted_costs]
            
def colored_plot_cb(x_nd, y_md, x_color, y_color, f, s_cloud_id, plot_color=1, proj_2d=1, x_labels=None, y_labels=None, label_colors=None):
    z_intercept = np.mean(x_nd[:,2])
    # Plot with color                                                       
    if plot_color:                                                          
        plotting_plt_shhuang.plot_tps_registration(x_nd, y_md, f, res = (.1, .1, .12), x_color = x_color, y_color = y_color, proj_2d=proj_2d, z_intercept=z_intercept, \
                                                   x_labels=x_labels, y_labels=y_labels, label_colors=label_colors)
    else:                                                                   
        plotting_plt_shhuang.plot_tps_registration(x_nd, y_md, f, res = (.3, .3, .12), proj_2d=proj_2d, z_intercept=z_intercept)
    # save plot to file                                                     
    #if use_vis == 0: vis = 'novis'                                          
    #elif use_vis == 1: vis = 'withcolor'                                    
    #elif use_vis == 2: vis = 'withlearnedlabels'                            
    #else: vis = 'withfeat'                                                  
    #if output_prefix is not None:                                           
    #    plt.savefig(output_prefix + '_' + s_cloud_id + '_' + vis + '.png')  

def get_alexnet_from_info_folder(net_info_folder):
    oldstdout_fno, oldstderr_fno = misc_util.suppress_stdout()

    net_prototxt = os.path.join(net_info_folder, 'alexnet_deploy.prototxt') 
    net_model = os.path.join(net_info_folder, 'alexnet_train_iter_10000')   
    net_mean = os.path.join(net_info_folder, 'mean.npy')

    net = get_alexnet(net_prototxt, net_model, net_mean)
    #unsuppress_stdout(_stderr, _stdout)
    misc_util.unsuppress_stdout(oldstdout_fno, oldstderr_fno)
    return net

###################

class Globals:
    robot = None
    env = None
    pr2 = None
    demo_clouds = {}  # Cache for demo point cloud info

def pr2_goto_start_posture():
    Globals.pr2.head.set_pan_tilt(0,1.15)
    Globals.pr2.rarm.goto_posture('side')
    Globals.pr2.larm.goto_posture('side')            
    Globals.pr2.join_all()
    Globals.pr2.update_rave()

def setup():
    trajoptpy.SetInteractive(args.interactive)

    rospy.init_node("exec_task",disable_signals=True)
    Globals.pr2 = PR2.PR2()
    Globals.env = Globals.pr2.env
    Globals.robot = Globals.pr2.robot
    Globals.viewer = trajoptpy.GetViewer(Globals.env)

    if args.soft_limits:
        Globals.pr2._set_rave_limits_to_soft_joint_limits()

def save_test_cloud(new_xyz, new_xyz_full, rgb, depth, T_w_k, alexnet_features):
    # Save new_xyz_ds, new_xyz_full, rgb, depth, T_w_k
    saved_test_clouds = h5py.File(TESTCLOUDS_F, 'r+');
    saved_test_clouds_g = saved_test_clouds.create_group(time.strftime("%d-%m-%Y_") + time.strftime("%H:%M:%S"))
    saved_test_clouds_g['cloud_xyz_ds'] = new_xyz
    saved_test_clouds_g['cloud_xyz_full'] = new_xyz_full
    saved_test_clouds_g['rgb'] = rgb
    saved_test_clouds_g['depth'] = depth
    saved_test_clouds_g['T_w_k'] = T_w_k
    saved_test_clouds_g['predicts'] = alexnet_features[0]
    saved_test_clouds_g['labels'] = alexnet_features[1]
    for f_key in alexnet_features[2]:
        saved_test_clouds_g['features_'+f_key] = alexnet_features[2][f_key]
    saved_test_clouds_g['valid_mask'] = alexnet_features[3]
    saved_test_clouds.close()


def exclude_gripper_finger_collisions():
    # This function is adapted from lfd/environment/simulation.py
    if not Globals.robot:
        return
    cc = trajoptpy.GetCollisionChecker(Globals.env)
    for lr in 'lr':
        for flr in 'lr':
            finger_link_name = "%s_gripper_%s_finger_tip_link" % (lr, flr)
            finger_link = Globals.robot.GetLink(finger_link_name)
            ipy.embed()
            for link in Globals.env.GetKinBody('table').GetLinks():
                cc.ExcludeCollisionPair(finger_link, link)

def main():
    setup()
    pr2_goto_start_posture()
    demofile = h5py.File(args.h5file, 'r')
    grabber = cloudprocpy.CloudGrabber()
    grabber.startRGBD()

    #exclude_gripper_finger_collisions()

    net = None
    if args.use_vis:  # load alexnet model
        print "Loading Alexnet model"
        net = get_alexnet_from_info_folder(args.net_info_folder)

    demos_saved_alexnet = None
    if os.path.isfile(args.h5file[:-3]+POSTFIX):
        demos_saved_alexnet = h5py.File(args.h5file[:-3]+POSTFIX, 'r')
    else:
        print "Could not load saved Alexnet demo features"

    subprocess.Popen(['rosrun', 'joint_states_listener_arms', 'joint_states_listener_arms.py'])

    prior_fn = None
    if args.use_vis:
        prior_fn = lambda demo_state, test_state: vis_utils.vis_cost_fn(demo_state, test_state, beta=BETA)
    tps_rpm_factory = TpsRpmRegistrationFactory(Globals.demo_clouds, n_iter=10, reg_init=10, reg_final=0.4, \
                                   rot_reg=np.r_[1e-4, 1e-4, 1e-1], \
                                   rad_init=0.01, rad_final = 0.00005, outlierprior=0.01, outlierfrac=0.01, prior_fn=prior_fn)
    while True:
        pr2_goto_start_posture()
        redprint("Acquire point cloud")
            
        rgb, depth = grabber.getRGBD()
        T_w_k = berkeley_pr2.get_kinect_transform(Globals.robot)
        new_xyz_full = cloud_proc_func(rgb, depth, T_w_k, returnOnlyXyz=True)

        (new_xyz, new_xyz_100corners, alexnet_features, alexnet_features_100corners) = vis_utils.get_alexnet_features(new_xyz_full, rgb, T_w_k, net, DS_SIZE, args)

        test_state = VisFeaturesSceneState(new_xyz, new_xyz_100corners, new_xyz_full, alexnet_features, alexnet_features_100corners)

        if args.extra_corners:
            new_xyz = new_xyz_100corners
            alexnet_features = alexnet_features_100corners
        save_test_cloud(new_xyz, new_xyz_full, rgb, depth, T_w_k, alexnet_features)

        ################################    

        redprint("Finding closest demonstration")

        success = False
        try_num = 0

        if not args.select_manual:
            seg_names_indices = find_closest_auto(test_state, demofile, demos_saved_alexnet, net, tps_rpm_factory)

        while not success:
            if try_num > 0:
                pr2_goto_start_posture()
            if args.select_manual:
                (seg_name, seg_idx) = find_closest_manual(demofile)
            else:  # Only works with execution on the robot
                (seg_name, seg_idx) = seg_names_indices[try_num]
            try_num += 1

            redprint("closest demo: %s"%(seg_name))
            if "done" in seg_name:
                redprint("DONE!")
                exit(0)
    
            ################################

            redprint("Generating end-effector trajectory")    

            demo_state = get_demo_state(demofile, seg_name, demos_saved_alexnet, net)
            old_xyz = np.concatenate((demo_state.scene_state.cloud, demo_state.scene_state.color), axis=1)
            old_xyz_full = np.concatenate((demo_state.scene_state.full_cloud, demo_state.scene_state.full_color), axis=1)

            handles = []
            handles.append(Globals.env.plot3(old_xyz_full[:,:3], 2, (1,0,0,1)))
            handles.append(Globals.env.plot3(new_xyz_full[:,:3], 2, (0,0,1,1)))

            print "Number of extra points:", len(get_extra_points())
            # Add extra points along the z axis; extra_old and extra_new should have same number of points
            extra_old = np.array([old_xyz.mean(axis=0)+[i,j,k,0,0,0] for (i,j,k) in get_extra_points()])
            extra_new = np.array([new_xyz.mean(axis=0)+[i,j,k,0,0,0] for (i,j,k) in get_extra_points()])

            redprint("Starting TPS-RPM")
            print "Original shape of demo: ", old_xyz.shape
            print "Original shape of test: ", new_xyz.shape
            tps_rpm_factory.demos = Globals.demo_clouds
            tps_rpm_factory.n_iter = 20

            demo_state_with_extra_pts = VisFeaturesSceneState(np.r_[old_xyz, extra_old], None, old_xyz_full, Globals.demo_clouds[seg_name].scene_state.alexnet_features, None)
            demo_object = Demonstration(seg_name, demo_state_with_extra_pts, None)

            test_state_with_extra_pts = VisFeaturesSceneState(np.r_[new_xyz, extra_new], None, new_xyz_full, test_state.alexnet_features, None)
            tps_rpm_f = tps_rpm_factory.register(demo_object, test_state_with_extra_pts)

            label_colors = ['g', 'c', 'b', 'y', 'r', 'k', 'm', '0.5', '#7fff00', '#b8860b']
            valid_old_xyz = demo_state.scene_state.get_valid_xyzrgb_cloud()
            valid_new_xyz = test_state.get_valid_xyzrgb_cloud()
            colored_plot_cb(valid_old_xyz[:,:-3], valid_new_xyz[:,:-3], \
                   valid_old_xyz[:,-3:], vis_utils.bgr_to_rgb(valid_new_xyz[:,-3:]), tps_rpm_f.f, \
                   seg_name, x_labels=demo_state_with_extra_pts.alexnet_features[0], y_labels=test_state.alexnet_features[0], label_colors=label_colors)

            success = exec_util.execute_traj(Globals, seg_name,  demofile[seg_name], handles, tps_rpm_f.f, old_xyz[:,:-3], new_xyz[:,:-3], args)

def get_extra_points():
    z_axis = [(0,0,1), (0,0,1.5)]
    x_axis = [(1,0,0), (-1,0,0), (0.75,0,0), (-0.75,0,0)]
    y_axis = [(0,1,0), (0,-1,0), (0,0.75,0), (0,-0.75,0)]
    corners_yz = [(0,1,1), (0,-1,1)]
    corners_xz = [(1,0,1), (-1,0,1)]
    return z_axis + x_axis + y_axis + corners_yz + corners_xz

if __name__ == "__main__":
    main()
