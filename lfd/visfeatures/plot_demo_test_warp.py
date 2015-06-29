#!/usr/bin/env python

import argparse, h5py, numpy as np, os
import scipy.spatial.distance as ssd
import openravepy, trajoptpy, time
from rapprentice import plotting_openrave
import misc_util, vis_utils
from alexnet_helper import get_alexnet
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd_execution import get_alexnet_from_info_folder, get_demo_state, get_extra_points
from lfd.demonstration.demonstration import Demonstration, VisFeaturesSceneState
from lfd.rapprentice.util import redprint
import matplotlib
import matplotlib.pyplot as plt
from lfd.rapprentice import plotting_plt

TEST_CLOUD_FILE = "saved_test_clouds_demsel.h5"
DEMO_FILE = "/home/shhuang/research/data_towel/towel_folding_demos_all_02-24-2015.h5"
NET_INFO_FOLDER = "/home/shhuang/research/data_towel/twoconv-threefc-towel-64-4"
POSTFIX = "saved_2conv3fc4_labels.h5"
DS_SIZE = 0.025
BETA = 10
TRANS_MATRIX = np.array([[0,1,0],[-1,0,0],[0,0,1]])
#VIEWER_CAMERA_MANIP_MATRIX = np.array([[ 0.02849845, -0.99890848,  0.03700917,  0.        ],
#       [ 0.99870579,  0.03001391,  0.04105977,  0.        ],
#       [-0.04212574,  0.03579113,  0.99847104,  0.        ],
#       [ 0.56596271,  0.06566502,  3.25062046,  1.        ]])
VIEWER_CAMERA_MANIP_MATRIX = np.array([[-0.02589365,  0.99893175, -0.0382736 ,  0.        ],
       [-0.98860062, -0.03126877, -0.14727888,  0.        ],
       [-0.14831832,  0.03402372,  0.98835422,  0.        ],
       [ 0.14395038,  0.05864128,  3.21041593,  1.        ]])
LABEL_COLORS_X = ['y', 'b', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']
LABEL_COLORS_Y = ['0.5', 'c', '0.5', '0.5', '0.5', '0.5', '0.5', '0.5', '0.5', '0.5']
counter = 0
prev_pts = None
prev_lines = None
prev_text = None

def get_objective_withviscost(prior_nm, x_nd, y_md, corr_nm, f):
    """
    Returns the following 6 objectives
    1/n \sum{i=1}^n \sum{j=1}^m corr_nm_ij ||y_md_j - f(x_nd_i)||_2^2
    bend_coef tr(w_ng' K_nn w_ng)
    tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
    rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij log corr_nm_ij -- NOT THIS
    -rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij -- NOT THIS
    1/n \sum{i=1}^n \sum{j=1}^m corr_nm_ij (beta c(x_nd_i, y_md_j))

    Parameter prior_nm already has coefficient beta multiplied in, but
    is not exponentiated
    """
    cost = np.zeros(6)
    dist_nm = ssd.cdist(x_nd, y_md, 'sqeuclidean')
    cost[0] = (corr_nm * dist_nm).sum() / len(x_nd)
    cost[1:3] = f.get_objective()[1:]
    cost[5] = (corr_nm * prior_nm).sum() / len(x_nd)
        
    corr_nm = np.reshape(corr_nm, (1,-1))
    nz_corr_nm = corr_nm[corr_nm != 0]
    #cost[3] = rad * (nz_corr_nm * np.log(nz_corr_nm)).sum()
    #cost[4] = -rad * nz_corr_nm.sum()
    return (cost[0], cost[1], cost[5])

def plot_callback(prior_nm, x_labels, label_colors, valid_x_nd, valid_y_md, x_nd, y_md, xtarg_nd, corr_nm, wt_n, f, rad):
    z_intercept = np.mean(x_nd[:,2])
    costs = get_objective_withviscost(prior_nm, x_nd, y_md, corr_nm, f)
    plot_tps_registration_proj_2d(costs, valid_x_nd[:,:3], valid_y_md[:,:3], f, (.05, .05, .05), z_intercept, x_labels, label_colors)

def plot_tps_registration_proj_2d(costs, x_nd, y_md, f, res, z_intercept, x_labels, label_colors):
    global counter, prev_pts, prev_lines, prev_text

    if prev_pts != None and prev_lines != None:
        time.sleep(0.5)
        prev_pts.remove()
        prev_lines.remove()
        for p in prev_text:
            p.remove()

    x_colors = []
    for i in range(len(x_nd)):
        label = x_labels[i]
        x_colors.append(label_colors[label])
    
    xwarped_nd = f.transform_points(x_nd)
    prev_pts = plt.scatter(xwarped_nd[:,0], xwarped_nd[:,1], edgecolors=x_colors, facecolors='none', marker='o', s=50)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    x_median = np.median(x_nd, axis=0)
    print costs
    prev_lines = plotting_plt.plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    prev_text = [0]*3
    prev_text[0] = plt.text(0.72, 0.15,'xyz cost: %.2e' % costs[0], fontsize=20, horizontalalignment='center', verticalalignment='center', transform=plt.axes().transAxes)
    prev_text[1] = plt.text(0.72, 0.05,'bending cost: %.2e' % costs[1], fontsize=20, horizontalalignment='center', verticalalignment='center', transform=plt.axes().transAxes)
    prev_text[2] = plt.text(0.72, 0.25,'rgb cost: %.2f' % costs[2], fontsize=20, horizontalalignment='center', verticalalignment='center', transform=plt.axes().transAxes)
    plt.pause(0.0001)

def apply_trans_matrix(A):
    return np.transpose(np.dot(TRANS_MATRIX, np.transpose(A)))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_cloud_name", type=str)
    parser.add_argument("demo_name", type=str)
    parser.add_argument("--no_vis", action="store_true")
    args = parser.parse_args()
    args.use_vis = 1
    args.extra_corners = 1
    args.downsample = 1

    testfile = h5py.File(TEST_CLOUD_FILE, 'r')
    demofile = h5py.File(DEMO_FILE, 'r')
    test = testfile[args.test_cloud_name]
    demo = demofile[args.demo_name]

    print "Loading CNN"
    net = get_alexnet_from_info_folder(NET_INFO_FOLDER, deploy_file="small_cnn_deploy.prototxt", model_file="towelcnn_train_iter_10000", small_cnn=True)
    demos_saved_alexnet = None
    if os.path.isfile(DEMO_FILE[:-3]+POSTFIX):
        demos_saved_alexnet = h5py.File(DEMO_FILE[:-3]+POSTFIX, 'r')
    else:
        print "Could not load saved CNN demo features"

    print "Getting appearance features"
    (test_xyz, test_xyz_100corners, alexnet_features, alexnet_features_100corners) = vis_utils.get_alexnet_features(test['cloud_xyz_full'][()], test['rgb'][()], test['T_w_k'][()], net, DS_SIZE, args)
    test_xyz[:,:3] = apply_trans_matrix(test_xyz[:,:3])
    test_xyz_100corners[:,:3] = apply_trans_matrix(test_xyz_100corners[:,:3])

    prior_fn = lambda demo_state, test_state: vis_utils.vis_cost_fn(demo_state, test_state, beta=BETA)
    tps_rpm_prior_fn = None
    if not args.no_vis:
        tps_rpm_prior_fn = prior_fn

    tps_rpm_factory = TpsRpmRegistrationFactory(None, n_iter=20, reg_init=0.1, reg_final=0.001, \
                                                rot_reg=np.r_[1e-4, 1e-4, 1e-1], \
                                                rad_init=0.01, rad_final = 0.00005, outlierprior=1e-3, outlierfrac=1e-3, prior_fn=tps_rpm_prior_fn)

    demo_state = get_demo_state(demofile, args.demo_name, demos_saved_alexnet, net, args)
    demo_state.scene_state.cloud[:,:3] = apply_trans_matrix(demo_state.scene_state.cloud[:,:3])
    #demo_state.scene_state.cloud_wcorners[:,:3] = apply_trans_matrix(demo_state.scene_state.cloud_wcorners[:,:3])
    #demo_state.scene_state.cloud_wocorners[:,:3] = apply_trans_matrix(demo_state.scene_state.cloud_wocorners[:,:3])
    demo_state.scene_state.full_cloud[:,:3] = apply_trans_matrix(demo_state.scene_state.full_cloud[:,:3])

    #import IPython as ipy
    #ipy.embed()
    old_xyz = np.concatenate((demo_state.scene_state.cloud, demo_state.scene_state.color), axis=1)
    old_xyz_full = np.concatenate((demo_state.scene_state.full_cloud, demo_state.scene_state.full_color), axis=1)

    print "Number of extra points:", len(get_extra_points())
    # Add extra points along the z axis; extra_old and extra_new should have same number of points
    extra_old = np.array([old_xyz.mean(axis=0)+[i,j,k,0,0,0] for (i,j,k) in get_extra_points()])
    extra_new = np.array([test_xyz_100corners.mean(axis=0)+[i,j,k,0,0,0] for (i,j,k) in get_extra_points()])

    redprint("Starting TPS-RPM")
    print "Original shape of demo: ", old_xyz.shape
    print "Original shape of test: ", test_xyz_100corners.shape

    demo_state_with_extra_pts = VisFeaturesSceneState(np.r_[old_xyz, extra_old], None, old_xyz_full, demo_state.scene_state.alexnet_features, None)
    demo_object = Demonstration(args.demo_name, demo_state_with_extra_pts, None)

    test_state_with_extra_pts = VisFeaturesSceneState(np.r_[test_xyz_100corners, extra_new], None, test['cloud_xyz_full'][()], alexnet_features_100corners, None)

    #prior_nm = None
    #if not args.no_vis:
    prior_nm, _ = prior_fn(demo_object.scene_state, test_state_with_extra_pts)

    valid_x_nd = demo_state.scene_state.get_valid_xyzrgb_cloud()
    test_state = VisFeaturesSceneState(test_xyz, test_xyz_100corners, test['cloud_xyz_full'][()], alexnet_features, alexnet_features_100corners)
    valid_y_md = test_state.get_valid_xyzrgb_cloud()

    plt.ion()
    fig = plt.figure('2d projection plot', figsize=(8.0, 4.5))
    fig.clear()

    y_labels = alexnet_features_100corners[0]
    y_colors = []
    for i in range(len(valid_y_md)):
        label = y_labels[i]
        y_colors.append(LABEL_COLORS_Y[label])
    plt.scatter(valid_y_md[:,0], valid_y_md[:,1], c=y_colors, edgecolors=y_colors, marker='+', s=50)
    vis_text = "with appearance information"
    if args.no_vis:
        vis_text = "without appearance information"
    plt.text(0.5, 0.92,vis_text, fontsize=24, horizontalalignment='center', verticalalignment='center', transform=plt.axes().transAxes)
    axes = plt.axes()
    axes.autoscale(enable=False)
    plt.pause(0.001)

    raw_input("Press ENTER when ready to continue")
    tps_rpm_f = tps_rpm_factory.register(demo_object, test_state_with_extra_pts, plotting=1, plot_cb=lambda a,b,c,d,e,f,g: plot_callback(prior_nm, demo_state.scene_state.alexnet_features[0],LABEL_COLORS_X,valid_x_nd,valid_y_md,a,b,c,d,e,f,g))

    plt.show()

if __name__ == "__main__":
    main()
