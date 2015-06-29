#!/usr/bin/env python

import argparse, h5py, os, pickle, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
import IPython as ipy

from lfd.demonstration.demonstration import Demonstration, VisFeaturesSceneState
from lfd.registration.registration import TpsRpmRegistrationFactory
import plotting_plt_shhuang, vis_utils

BETA = 10
WARP_FNS_FILE = 'warp_fn_and_costs.p'

def random_split(all_clouds_file, source_clouds_file, target_clouds_file, \
                 frac_source=0.8):
    all_clouds = h5py.File(all_clouds_file, 'r')
    num_source_clouds = int(frac_source * len(all_clouds))
    indices = range(len(all_clouds))
    source_cloud_ids = set(random.sample(indices, num_source_clouds))
    print source_cloud_ids

    source_clouds = h5py.File(source_clouds_file, 'w')
    target_clouds = h5py.File(target_clouds_file, 'w')

    for i in range(len(all_clouds)):
        cloud_id = all_clouds.keys()[i]
        print cloud_id
        if i in source_cloud_ids:
            all_clouds.copy(cloud_id, source_clouds)
        else:
            all_clouds.copy(cloud_id, target_clouds)
    all_clouds.close()
    source_clouds.close()
    target_clouds.close()

def plot_cb_gen(output_prefix, y_color, plot_color=1, proj_2d=1):
    def plot_cb(x_nd, y_md, x_color, f, s_cloud_id, use_vis, x_labels=None, y_labels=None, label_colors=None):
        z_intercept = np.mean(x_nd[:,2])
        plotting_plt_shhuang.plot_tps_registration(x_nd, y_md, f, res = (.1, .1, .12), x_color = x_color, y_color = y_color, proj_2d=proj_2d, z_intercept=z_intercept, \
                x_labels=x_labels, y_labels=y_labels, label_colors=label_colors)
        # save plot to file
        if use_vis == 0: vis = 'novis'
        elif use_vis == 1: vis = 'withlearnedlabels'
        else: vis = 'ERROR'
        if output_prefix is not None:
            plt.savefig(output_prefix + '_' + s_cloud_id + '_' + vis + '.png')
    return plot_cb

def find_tps_rpm_warp(test_state, t_cloud_id, plot_cb, demo_objects, tps_rpm_factory, tps_rpm_factory_novis):
    f_and_costs_with_st_ids = []
    label_colors = ['g', 'c', 'b', 'y', 'r', 'k', 'm', '0.5', '#7fff00', '#b8860b']
    for (i,k) in enumerate(demo_objects):
        demo_object = demo_objects[k]
        tps_rpm_f = tps_rpm_factory.register(demo_object, test_state)

        tps_rpm_f_novis = tps_rpm_factory_novis.register(demo_object, test_state)
        label_predicts_demo = demo_object.scene_state.alexnet_features[0]
        label_predicts_test = test_state.alexnet_features[0]
        f_and_costs = [(tps_rpm_f.f, tps_rpm_f.get_objective()[1], 1, label_predicts_demo, label_predicts_test),
                       (tps_rpm_f_novis.f, tps_rpm_f_novis.get_objective()[1], 0, label_predicts_demo, label_predicts_test)]

        for j in range(len(f_and_costs)):
            if plot_cb != None:
                plot_cb(demo_object.scene_state.get_valid_xyzrgb_cloud()[:,:-3], test_state.get_valid_xyzrgb_cloud()[:,:-3], \
                        vis_utils.bgr_to_rgb(demo_object.scene_state.get_valid_xyzrgb_cloud()[:,-3:]), f_and_costs[j][0], \
                        demo_object.name, f_and_costs[j][2], x_labels=f_and_costs[j][3], y_labels=f_and_costs[j][4], label_colors=label_colors)
            f_and_costs_with_st_ids.append((f_and_costs[j][2], demo_object.name, t_cloud_id, \
                                            f_and_costs[j][0], f_and_costs[j][1]))
    return f_and_costs_with_st_ids

def get_alexnet_features_from_seg(seg):
    features = {}
    features = {}
    features['fc7'] = seg['fc7_ds'][()]
    features['fc8'] = seg['fc8_ds'][()]
    return [seg['predicts_ds'][()], seg['scores_ds'][()], features, seg['validmask_ds'][()]]

def get_demo_state(cloud_id, clouds_f):
    seg = clouds_f[cloud_id]
    cloud_xyz = seg['cloud_xyz_full'][()]
    cloud_xyz_ds = seg['cloud_xyz_ds'][()]
    alexnet_features = get_alexnet_features_from_seg(seg)
    visfeatures_scene = VisFeaturesSceneState(cloud_xyz_ds, cloud_xyz_ds, cloud_xyz, alexnet_features, alexnet_features)
    return Demonstration(cloud_id, visfeatures_scene, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_clouds_file", type=str)
    parser.add_argument("target_clouds_file", type=str)
    parser.add_argument("output_folder", type=str)
    args = parser.parse_args()

    source_clouds_f = h5py.File(args.source_clouds_file, 'r')
    demo_objects = {}
    for k in source_clouds_f.keys():
        demo_objects[k] = get_demo_state(k, source_clouds_f)

    target_clouds_f = h5py.File(args.target_clouds_file, 'r')
    target_ids = target_clouds_f.keys()

    prior_fn = lambda demo_state, test_state: vis_utils.vis_cost_fn(demo_state, test_state, beta=BETA)
    tps_rpm_factory = TpsRpmRegistrationFactory(demo_objects, n_iter=10, reg_init=0.1, reg_final=0.001, \
                                   rot_reg=np.r_[1e-4, 1e-4, 1e-1], \
                                   rad_init=0.01, rad_final = 0.00005, outlierprior=1e-3, outlierfrac=1e-3, prior_fn=prior_fn)
    tps_rpm_factory_novis = TpsRpmRegistrationFactory(demo_objects, n_iter=10, reg_init=0.1, reg_final=0.001, \
                                   rot_reg=np.r_[1e-4, 1e-4, 1e-1], \
                                   rad_init=0.01, rad_final = 0.00005, outlierprior=1e-3, outlierfrac=1e-3, prior_fn=None)

    # Visualize warp (and save warp png, warp f + warp cost) for each pair of
    # source and target point clouds
    warp_fns_and_costs = []

    targets_seen = set(["crumple02", "crumple08", "crumple12", "flat_towel11", "flat_towel12", "flat_towel15", "flat_towel1", "flat_towel3", "flat_towel6"])

    for i_t in range(len(target_ids)):
        print i_t, target_ids[i_t]
        if target_ids[i_t] in targets_seen:
            continue
        t_cloud_id = target_ids[i_t]
        g = target_clouds_f[t_cloud_id]
        test_alexnet_features = get_alexnet_features_from_seg(g)
        test_state = VisFeaturesSceneState(g['cloud_xyz_ds'][()], g['cloud_xyz_ds'][()],
                                           g['cloud_xyz_ds'][()], test_alexnet_features, test_alexnet_features)

        plot_fn = None
        if target_ids[i_t].find('relay') >= 0 or target_ids[i_t].find('triangle') >= 0:
            plot_fn = plot_cb_gen(os.path.join(args.output_folder, t_cloud_id), \
                                  vis_utils.bgr_to_rgb(test_state.get_valid_xyzrgb_cloud()[:,-3:]))


        # Find TPS-RPM with specified visual information
        warp_fns_and_costs += find_tps_rpm_warp(test_state, t_cloud_id, plot_fn, demo_objects,
                                                tps_rpm_factory, tps_rpm_factory_novis)

        # Save warp_fns_and_costs file at intermediate steps
        with open(os.path.join(args.output_folder, WARP_FNS_FILE), 'wb') as g:
            pickle.dump(warp_fns_and_costs, g)
            
if __name__ == "__main__":
    main()
