#!/usr/bin/env python

# Precalculates the following between each test scene and demo:
#     1) KL divergence for patch categories
#     2) matching cost for registering point clouds (w/o visual features, after downsampling and before adding extra corners)
#            1/n \sum{i=1}^n \sum{j=1}^m corr_nm_ij ||y_md_j - f(x_nd_i)||_2^2
#     3) bending cost for registering point clouds (w/o visual features, after downsampling and before adding extra corners)
#            bend_coef tr(w_ng' K_nn w_ng)
#     4) rotational penalty
#            tr((lin_ag - I) diag(rot_coef) (lin_ag - I))
#     5) rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij log corr_nm_ij
#     6) -rad \sum{i=1}^n \sum{j=1}^m corr_nm_ij

import argparse, h5py, os, numpy as np
import vis_utils, misc_util
from alexnet_helper import get_alexnet
from lfd.demonstration.demonstration import Demonstration, VisFeaturesSceneState
from lfd.registration.registration import TpsRpmRegistrationFactory

DS_SIZE = .025
POSTFIX = 'saved_alexnet_labels.h5' 

def get_alexnet_from_info_folder(net_info_folder):
    oldstdout_fno, oldstderr_fno = misc_util.suppress_stdout()

    net_prototxt = os.path.join(net_info_folder, 'alexnet_deploy.prototxt') 
    net_model = os.path.join(net_info_folder, 'alexnet_train_iter_10000')   
    net_mean = os.path.join(net_info_folder, 'mean.npy')

    net = get_alexnet(net_prototxt, net_model, net_mean)
    misc_util.unsuppress_stdout(oldstdout_fno, oldstderr_fno)
    return net

def get_demo_state(demofile, cloud_id, demos_saved_alexnet, net):
    # Returns a Demonstration object with VisFeaturesSceneState (alexnet_features = None if use_vis == False)
    if demos_saved_alexnet != None:  # Clouds already have RGB, not BGR
        seg = demos_saved_alexnet[cloud_id]
        cloud_xyz = seg['cloud_xyz_full'][()]

        cloud_xyz_ds = seg['cloud_xyz_ds'][()]
        cloud_xyz_ds_100corners = None

        alexnet_features = None
        alexnet_features_100corners = None
        features = {}
        features['fc7'] = seg['fc7_ds'][()]
        features['fc8'] = seg['fc8_ds'][()]
        alexnet_features = [seg['predicts_ds'][()], seg['scores_ds'][()], features, seg['validmask_ds'][()]]

    else:
        seg = demofile[cloud_id]
        redprint("Calculating Alexnet for demo")
        cloud_xyz = np.asarray(seg['cloud_xyz'][()])
        (cloud_xyz_ds, cloud_xyz_ds_100corners, alexnet_features, alexnet_features_100corners) = vis_utils.get_alexnet_features(cloud_xyz, seg['rgb'][()], seg['T_w_k'][()], net, DS_SIZE, args, use_vis=True)

    visfeatures_scene = VisFeaturesSceneState(cloud_xyz_ds, cloud_xyz_ds_100corners, cloud_xyz, alexnet_features,
                                                  alexnet_features_100corners)
    return Demonstration(cloud_id, visfeatures_scene, None)

def load_test_cloud(testh5file, test_k):
    g = testh5file[test_k]
    return (g['rgb'], g['depth'], g['T_w_k'], g['cloud_xyz_full'])

def registration_cost(demo_info, test_state, tps_rpm_factory=None):
    # demo_info: a Demonstration with VisFeatureSceneState
    # test_state: a VisFeatureSceneState
    if tps_rpm_factory == None:
        tps_rpm_factory = Globals.tps_rpm_factory

    reg = tps_rpm_factory.register(demo_info, test_state)
    return reg.get_objective()

def get_features(test_state, demo_clouds, net, tps_rpm_factory):
    demo_keys = list(demo_clouds.keys())
    chisq_dist = [vis_utils.get_labels_distance(test_state.alexnet_features_wocorners[0], \
                                                demo_clouds[k].scene_state.alexnet_features_wocorners[0]) \
                  for k in demo_keys]
    label_pairs_dist = [vis_utils.get_label_pairs_distance(test_state.alexnet_features_wocorners[0], \
                                       demo_clouds[k].scene_state.alexnet_features_wocorners[0])
                        for k in demo_keys]

    costs = [tuple(registration_cost(demo_clouds[k], test_state, tps_rpm_factory)) for k in demo_keys]
   
    return [costs[i] + (chisq_dist[i],) + label_pairs_dist[i] for i in range(len(chisq_dist))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demoh5file", type=str)
    parser.add_argument("testh5file", type=str)
    parser.add_argument("net_info_folder", type=str)  # folder that contains alexnet_deploy.prototxt, alexnet_train_iter_10000, mean.npy
    parser.add_argument("outputfile", type=str)

    parser.add_argument("--cloud_proc_func", default="extract_nongreen")
    parser.add_argument("--cloud_proc_mod", default="towel_color_filters")
    
    args = parser.parse_args()
    args.use_vis = False
    args.downsample = True
    args.extra_corners = False

    demofile = h5py.File(args.demoh5file, 'r')
    net = None

    print "Loading Alexnet model"
    net = get_alexnet_from_info_folder(args.net_info_folder)

    print "Loading saved Alexnet demo features"
    demos_saved_alexnet = None
    if os.path.isfile(args.demoh5file[:-3]+POSTFIX):
        demos_saved_alexnet = h5py.File(args.demoh5file[:-3]+POSTFIX, 'r')
    else:
        print "Could not load saved Alexnet demo features"

    demo_clouds = {}  # Cache for demo point cloud info
    for k in demofile:  # Store all demos in Globals.demo_clouds
        demo_clouds[k] = get_demo_state(demofile, k, demos_saved_alexnet, net)

    tps_rpm_factory = TpsRpmRegistrationFactory(demo_clouds, n_iter=10, reg_init=10, reg_final=0.001, \
                                   rot_reg=np.r_[1e-4, 1e-4, 1e-1], \
                                   rad_init=0.01, rad_final = 0.00005, outlierprior=0.0001, outlierfrac=0.0001, prior_fn=None)

    testh5file = h5py.File(args.testh5file, 'r')
    outputf = h5py.File(args.outputfile, 'w')
    demo_keys = demofile.keys()

    count = 1
    for test_k in testh5file.keys():
        if count % 10 == 0:
            print count
        count += 1
        test_g = outputf.create_group(test_k)
        (rgb, depth, T_w_k, new_xyz_full) = load_test_cloud(testh5file, test_k)
        (new_xyz, new_xyz_100corners, alexnet_features, alexnet_features_100corners) = vis_utils.get_alexnet_features(new_xyz_full, rgb, T_w_k, net, DS_SIZE, args, use_vis=True)
        # new_xyz_100corners, alexnet_features_100corners should be None

        test_state = VisFeaturesSceneState(new_xyz, new_xyz_100corners, new_xyz_full, alexnet_features, alexnet_features_100corners)

        feature_list = get_features(test_state, demo_clouds, net, tps_rpm_factory)
        for i in range(len(feature_list)):
            test_g[demo_keys[i]] = feature_list[i]

    testh5file.close()
    demofile.close()
    outputf.close()


if __name__ == "__main__":
    main()
