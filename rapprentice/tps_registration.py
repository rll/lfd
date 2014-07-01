#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
from rapprentice import registration, math_utils
from rapprentice.registration import loglinspace, ThinPlateSpline, fit_ThinPlateSpline
import tps
import knot_classifier

import IPython as ipy

N_ITER_CHEAP = 14
EM_ITER_CHEAP = 1

def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def rgb2xyz(rgb):
    """
    r,g,b ranges from 0 to 1
    http://en.wikipedia.org/wiki/SRGB_color_space
    http://en.wikipedia.org/wiki/CIE_XYZ
    """
    rgb_linear = np.empty_like(rgb) # copy rgb so that the original rgb is not modified
    
    cond = rgb > 0.04045
    rgb_linear[cond] = np.power((rgb[cond] + 0.055) / 1.055, 2.4)
    rgb_linear[~cond] = rgb[~cond] / 12.92
    
    rgb_to_xyz = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]])
    xyz = rgb_linear.dot(rgb_to_xyz.T)
    return xyz

def xyz2lab(xyz):
    """
    l ranges from 0 to 100 and a,b ranges from -128 to 128
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    ref = np.array([0.95047, 1., 1.08883]) # CIE LAB constants for Observer = 2deg, Illuminant = D65
    xyz = xyz / ref # copy xyz so that the original xyz is not modified

    cond = xyz > 0.008856
    xyz[cond] = np.power(xyz[cond], 1./3.)
    xyz[~cond] = 7.787 * xyz[~cond] + 16./116.
    
    x,y,z = xyz.T
    l = 116. * y - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    
    lab = np.array([l,a,b]).T
    return lab

def ab_cost(xyzrgb1, xyzrgb2):
    _,d = xyzrgb1.shape
    d -= 3  # subtract out the three RGB coordinates
    lab1 = rgb2lab(xyzrgb1[:,d:])
    lab2 = rgb2lab(xyzrgb2[:,d:])
    cost = ssd.cdist(lab1[:,1:], lab2[:,1:], 'euclidean')
    return cost

def sinkhorn_balance_coeffs(prob_NM, normalize_iter):
    """
    Computes the coefficients to balance the matrix prob_NM. Similar to balance_matrix3. Column-normalization happens first.
    The coefficients are computed with type 'f4', so it's better if prob_NM is already in type 'f4'.
    The sinkhorn_balance_matrix can be then computed in the following way:
    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    """
    if prob_NM.dtype != np.dtype('f4'):
        prob_NM = prob_NM.astype('f4')
    N,M = prob_NM.shape
    r_N = np.ones(N,'f4')
    c_M = np.ones(M,'f4')
    for _ in xrange(normalize_iter):
        c_inv_M = r_N.dot(prob_NM)
        c_zero_M = c_inv_M < 1e-20
        c_M[~c_zero_M] = 1./c_inv_M[~c_zero_M] # normalize along columns
        c_M[c_zero_M] = 0.
        r_inv_N = prob_NM.dot(c_M)
        r_zero_N = r_inv_N < 1e-20
        r_N[~r_zero_N] = 1./r_inv_N[~r_zero_N] # normalize along rows
        r_N[r_zero_N] = 0.

    return r_N, c_M

def tps_rpm(x_nd, y_md, n_iter = 20, lambda_init = 10., lambda_final = .1, T_init = .04, T_final = .00004, rot_reg = np.r_[1e-4, 1e-4, 1e-1], 
            plotting = False, plot_cb = None, vis_cost_xy = None, outlierprior = 1e-1, outlierfrac = 1e-2, em_iter = 2, user_data=None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    lambda_init/lambda_final: regularization on curvature
    T_init/T_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    vis_cost_xy: matrix of pairwise costs between source and target points, based on visual features
    Note: Pick a T_init that is about 1/10 of the largest square distance of all point pairs
    """
    _,d=x_nd.shape
    lambdas = loglinspace(lambda_init, lambda_final, n_iter)
    Ts = loglinspace(T_init, T_final, n_iter)

    f = ThinPlateSpline(d)
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_nd,axis=0) - np.min(x_nd,axis=0))
    f.lin_ag = np.diag(scale).T # align the mins and max
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) * scale  # align the medians

    for i in xrange(n_iter):
        for _ in xrange(em_iter):
            f, corr_nm = rpm_em_step(x_nd, y_md, lambdas[i], Ts[i], rot_reg, f, vis_cost_xy = vis_cost_xy, outlierprior = outlierprior, outlierfrac = outlierfrac, user_data = user_data)

        if plotting and (i%plotting==0 or i==(n_iter-1)):
            plot_cb(x_nd, y_md, corr_nm, f, i)
    return f, corr_nm

def rpm_em_step(x_nd, y_md, l, T, rot_reg, prev_f, vis_cost_xy = None, outlierprior = 1e-1, outlierfrac = 1e-2, normalize_iter = 10, user_data=None):
    n,d = x_nd.shape
    m,_ = y_md.shape
    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
    prob_nm = np.exp( -dist_nm / (2*T) ) / np.sqrt(2 * np.pi * T) # divide by constant term so that outlierprior makes sense as a pr
    if vis_cost_xy != None:
        pi = np.exp( -vis_cost_xy )
        pi /= pi.max() # rescale the maximum probability to be 1. effectively, the outlier priors are multiplied by a visual prior of 1 (since the outlier points have a visual prior of 1 with any point)
        prob_nm *= pi
    
    x_priors = np.ones(n)*outlierprior    
    y_priors = np.ones(m)*outlierprior    
    corr_nm, r_N, _ =  registration.balance_matrix3(prob_nm, normalize_iter, x_priors, y_priors, outlierfrac)
    corr_nm += 1e-9
    
    f = fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, l, rot_reg)

    return f, corr_nm

def fit_ThinPlateSpline_corr(x_nd, y_md, corr_nm, l, rot_reg, x_weights = None):
    wt_n = corr_nm.sum(axis=1)

    if np.any(wt_n == 0):
        inlier = wt_n != 0
        x_nd = x_nd[inlier,:]
        wt_n = wt_n[inlier,:]
        x_weights = x_weights[inlier]
        xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
    else:
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)

    if x_weights is not None:
        if x_weights.ndim > 1:
            wt_n=wt_n[:,None]*x_weights
        else:
            wt_n=wt_n*x_weights
    
    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = l, wt_n = wt_n, rot_coef = rot_reg)
    f._bend_coef = l
    f._wt_n = wt_n
    f._rot_coef = rot_reg
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, l, wt_n=wt_n)/wt_n.mean()
    
    return f

def rpm_em_step_stat(x_nd, y_md, l, T, rot_reg, prev_f, vis_cost_xy = None, outlierprior = 1e-2, normalize_iter = 20, T0 = .04, user_data=None):
    """
    Statiscal interpretation of the RPM EM step
    """
    n,d = x_nd.shape
    m,_ = y_md.shape
    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
    outlier_dist_1m = ssd.cdist(xwarped_nd.mean(axis=0)[None,:], y_md, 'sqeuclidean')
    outlier_dist_n1 = ssd.cdist(xwarped_nd, y_md.mean(axis=0)[None,:], 'sqeuclidean')

    # Note: proportionality constants within a column can be ignored since Sinkorn balancing normalizes the columns first
    prob_nm = np.exp( -(dist_nm / (2*T)) + (outlier_dist_1m / (2*T0)) ) / np.sqrt(T) # divide by np.exp( outlier_dist_1m / (2*T0) ) to prevent prob collapsing to zero
    if vis_cost_xy != None:
        pi = np.exp( -vis_cost_xy )
        pi /= pi.sum(axis=0)[None,:] # normalize along columns; these are proper probabilities over j = 1,...,N
        prob_nm *= (1. - outlierprior) * pi
    else:
        prob_nm *= (1. - outlierprior) / float(n)
    outlier_prob_1m = outlierprior * np.ones((1,m)) / np.sqrt(T0) # divide by np.exp( outlier_dist_1m / (2*T0) )
    outlier_prob_n1 = np.exp( -outlier_dist_n1 / (2*T0) ) / np.sqrt(T0)
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m][:,None] = outlier_prob_n1
    prob_NM[n, :m][None,:] = outlier_prob_1m
    prob_NM[n, m] = 0
    
    r_N, c_M = sinkhorn_balance_coeffs(prob_NM, normalize_iter)
    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    # prob_NM needs to be row-normalized at this point
    corr_nm = prob_NM[:n, :m]
    
    wt_n = corr_nm.sum(axis=1)

    # discard points that are outliers (i.e. their total correspondence is smaller than 1e-2)    
    inlier = wt_n > 1e-2
    if np.any(~inlier):
        x_nd = x_nd[inlier,:]
        wt_n = wt_n[inlier,:]
        xtarg_nd = (corr_nm[inlier,:]/wt_n[:,None]).dot(y_md)
    else:
        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = l, wt_n = wt_n, rot_coef = rot_reg)
    f._bend_coef = l
    f._rot_coef = rot_reg
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, l, wt_n=wt_n)/wt_n.mean()

    return f, corr_nm

def calc_segment_corr(rope_nodes1, pts_segmentation_inds0, pts_segmentation_inds1):
    n = pts_segmentation_inds0[-1]
    m = pts_segmentation_inds1[-1]
    corr_nm = np.zeros((n, m))
    for i, (i_start0, i_end0, i_start1, i_end1) in enumerate(zip(pts_segmentation_inds0[:-1], pts_segmentation_inds0[1:], pts_segmentation_inds1[:-1], pts_segmentation_inds1[1:])):
        lengths = np.array([0])
        if i_end1-i_start1 > 1:
            heights = np.apply_along_axis(np.linalg.norm, 1, np.diff(rope_nodes1[i_start1:i_end1,:], axis=0))
            lengths = np.r_[lengths, heights]
        summed_lengths = np.cumsum(lengths)
        corr_nm[i_start0:i_end0,i_start1:i_end1] = math_utils.interp_mat(np.linspace(0, summed_lengths[-1], i_end0-i_start0), summed_lengths)
    return corr_nm

def tile(A, tile_pattern):
    B = np.zeros((A.shape[0]*tile_pattern.shape[0], A.shape[1]*tile_pattern.shape[1]))
    for i in range(tile_pattern.shape[0]):
        for j in range(tile_pattern.shape[1]):
            if tile_pattern[i,j]:
                B[A.shape[0]*i:A.shape[0]*(i+1), A.shape[1]*j:A.shape[1]*(j+1)] = A
    return B
            
def tps_segment_registration(rope_nodes_or_crossing_info0, rope_nodes_or_crossing_info1, cloud0 = None, cloud1 = None, corr_tile_pattern = np.array([[1]]), rev_perm = None,
                             x_weights = None, reg = .1, rot_reg = np.r_[1e-4, 1e-4, 1e-1], plotting = False, plot_cb = None):
    """
    Find a registration by assigning correspondences based on the topology of the rope
    If rope_nodes0 and rope_nodes1 have the same topology (up to a variant of removing the last crossing in open ropes), the correspondences are given by linearly interpolating segments of both rope_nodes. The rope_nodes are segmented based on crossings.
    If rope_nodes0 and rope_nodes1 don't have the same topology, this function returns None for the TPS and the correspondence matrix
    rope_nodes_or_crossing_info is either rope nodes, which is an ordered sequence of points (i.e. it is the back bone of its respective rope), or is a tuple containing the rope nodes and crossings information (the information returned by knot_classifier.calculateCrossings)
    rev_perm is the permutation matrix of how corr_tile_pattern changes when the rope_nodes have been reversed
    """
    if type(rope_nodes_or_crossing_info0) == tuple:
        rope_nodes0, crossings0, crossings_links_inds0, cross_pairs0, rope_closed0 = rope_nodes_or_crossing_info0
    else:
        rope_nodes0 = rope_nodes_or_crossing_info0
        crossings0, crossings_links_inds0, cross_pairs0, rope_closed0 = knot_classifier.calculateCrossings(rope_nodes0)
    if type(rope_nodes_or_crossing_info1) == tuple:
        rope_nodes1, crossings1, crossings_links_inds1, cross_pairs1, rope_closed1 = rope_nodes_or_crossing_info1
    else:
        rope_nodes1 = rope_nodes_or_crossing_info1
        crossings1, crossings_links_inds1, cross_pairs1, rope_closed1 = knot_classifier.calculateCrossings(rope_nodes1)

    n,d = rope_nodes0.shape
    m,_ = rope_nodes1.shape
    
    # Compile all possible (reasonable) registrations and later select the one with the lowest bending cost
    f_variations = []
    corr_nm_variations = []
    
    # Add registrations for the closed versions of any open rope
    if not rope_closed0 or not rope_closed1:
        rope_nodes_crossing_infos0 = []
        rope_nodes_crossing_infos1 = []
        if not rope_closed0:
            for end in [0,-1]:
                rope_nodes_crossing_infos0.append((rope_nodes0,) + knot_classifier.close_rope(crossings0, crossings_links_inds0, cross_pairs0, end) + (True,))
        else:
            rope_nodes_crossing_infos0.append((rope_nodes0, crossings0, crossings_links_inds0, cross_pairs0, True))
        if not rope_closed1:
            for end in [0,-1]:
                rope_nodes_crossing_infos1.append((rope_nodes1,) + knot_classifier.close_rope(crossings1, crossings_links_inds1, cross_pairs1, end) + (True,))
        else:
            rope_nodes_crossing_infos1.append((rope_nodes1, crossings1, crossings_links_inds1, cross_pairs1, True))
        for rope_nodes_crossing_info0 in rope_nodes_crossing_infos0:
            for rope_nodes_crossing_info1 in rope_nodes_crossing_infos1:
                f_var, corr_nm_var = tps_segment_registration(rope_nodes_crossing_info0, rope_nodes_crossing_info1, cloud0 = cloud0, cloud1 = cloud1, corr_tile_pattern = corr_tile_pattern, 
                                                              x_weights = x_weights, reg = reg, rot_reg = rot_reg, plotting = False, plot_cb = None)
                f_variations.append(f_var)
                corr_nm_variations.append(corr_nm_var)
    
    crossings0 = np.array(crossings0)
    crossings1 = np.array(crossings1)
    crossings_links_inds0 = np.array(crossings_links_inds0)
    crossings_links_inds1 = np.array(crossings_links_inds1)
    
    pts_segmentation_inds0 = np.r_[0, crossings_links_inds0 + 1, n]
    pts_segmentation_inds1 = np.r_[0, crossings_links_inds1 + 1, m]

    if cross_pairs0 == cross_pairs1: # same topology
        # need to try the tps registration f for rope_nodes1 and/or the reverse rope_nodes1
        reversed_rope_points1_variations = []
        if np.all(crossings0 == crossings1):
            reversed_rope_points1_variations.append(False)
        # could happen when (1) rope_nodes1 are in a reverse order compared to rope_nodes0, or (2) crossings1 is a palindrome, or (3) both
        if np.all(crossings0 == crossings1[::-1]):
            reversed_rope_points1_variations.append(True)
        
        if len(reversed_rope_points1_variations) > 0:
            for reversed_rope_points1 in reversed_rope_points1_variations:
                if reversed_rope_points1:
                    corr_nm_var = calc_segment_corr(rope_nodes1[::-1], pts_segmentation_inds0, m - pts_segmentation_inds1[::-1])
                    corr_nm_var = corr_nm_var[:,::-1]
                    if rev_perm is None:
                        rev_perm = np.eye(len(corr_tile_pattern))
                        rev_perm = rev_perm[::-1]
                        rev_perm = np.r_[rev_perm[(len(rev_perm)/2)-1:,:], rev_perm[:(len(rev_perm)/2)-1,:]]
                    corr_nm_var_aug = tile(corr_nm_var, rev_perm.dot(corr_tile_pattern))
                else:
                    corr_nm_var = calc_segment_corr(rope_nodes1, pts_segmentation_inds0, pts_segmentation_inds1)
                    corr_nm_var_aug = tile(corr_nm_var, corr_tile_pattern)
                
                cloud0_var = cloud0 if cloud0 is not None else rope_nodes0
                cloud1_var = cloud1 if cloud1 is not None else rope_nodes1
                assert corr_nm_var_aug.shape == (len(cloud0_var), len(cloud1_var))

                f_var = fit_ThinPlateSpline_corr(cloud0_var, cloud1_var, corr_nm_var_aug, reg, rot_reg, x_weights)
        
                f_variations.append(f_var)
                corr_nm_variations.append(corr_nm_var)
    
    # filter out the invalid registrations
    f_variations = [f_var for f_var in f_variations if f_var is not None]
    corr_nm_variations = [corr_nm_var for corr_nm_var in corr_nm_variations if corr_nm_var is not None]

    if not f_variations:
        f = None
        corr_nm = None
    else:
        if len(f_variations) > 1:
            reflected_reg_costs = [(np.linalg.det(f_var.lin_ag) < 0, registration.tps_reg_cost(f_var)) for f_var in f_variations] # first element indicates if the affine part of this transformation is a reflection
            # sort the registrations from non-reflected to reflected transformations first and then from low to high bending cost to break ties
            f_variations, corr_nm_variations = zip(*[(f_var, corr_nm_var) for (reflected_reg_cost, f_var, corr_nm_var) in sorted(zip(reflected_reg_costs, f_variations, corr_nm_variations))])
        f = f_variations[0]
        corr_nm = corr_nm_variations[0]
    
    # TODO plot correct pts_segmemtation_inds
    if plotting:
        corr_nm_aug = tile(corr_nm, corr_tile_pattern) if corr_nm is not None else None
        plot_cb(rope_nodes0, rope_nodes1, cloud0, cloud1, corr_nm, corr_nm_aug, f, pts_segmentation_inds0, pts_segmentation_inds1)

    return f, corr_nm

def main():
    import argparse, h5py, os
    import matplotlib.pyplot as plt
    from rapprentice import clouds, plotting_plt
    import registration
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--i_start", type=int, default=0)
    parser.add_argument("--i_end", type=int, default=-1)
    regtype_choices = ['rpm', 'rpm-cheap', 'rpm-bij', 'rpm-bij-cheap']
    parser.add_argument("--regtypes", type=str, nargs='*', choices=regtype_choices, default=regtype_choices)
    parser.add_argument("--plot_color", type=int, default=1)
    parser.add_argument("--proj", type=int, default=1, help="project 3d visualization into 2d")
    parser.add_argument("--visual_prior", type=int, default=1)
    parser.add_argument("--plotting", type=int, default=1)

    args = parser.parse_args()
    
    def plot_cb_gen(output_prefix, args, x_color, y_color):
        def plot_cb(x_nd, y_md, corr_nm, f, iteration):
            if args.plot_color:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, x_color = x_color, y_color = y_color, proj_2d=args.proj)
            else:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, proj_2d=args.proj)
            # save plot to file
            if output_prefix is not None:
                plt.savefig(output_prefix + "_iter" + str(iteration) + '.png')
        return plot_cb

    def plot_cb_bij_gen(output_prefix, args, x_color, y_color):
        def plot_cb_bij(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f):
            if args.plot_color:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.3, .3, .12), x_color = x_color, y_color = y_color, proj_2d=args.proj)
            else:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.4, .3, .12), proj_2d=args.proj)
            # save plot to file
            if output_prefix is not None:
                plt.savefig(output_prefix + "_iter" + str(iteration) + '.png')
        return plot_cb_bij

    # preprocess and downsample clouds
    DS_SIZE = 0.025
    infile = h5py.File(args.input_file)
    source_clouds = {}
    target_clouds = {}
    for i in range(args.i_start, len(infile) if args.i_end==-1 else args.i_end):
        source_cloud = clouds.downsample(infile[str(i)]['source_cloud'][()], DS_SIZE)
        source_clouds[i] = source_cloud
        target_clouds[i] = []
        for (cloud_key, target_cloud) in infile[str(i)]['target_clouds'].iteritems():
            target_cloud = clouds.downsample(target_cloud[()], DS_SIZE)
            target_clouds[i].append(target_cloud)
    infile.close()
    
    tps_costs = []
    tps_reg_costs = []
    for regtype in args.regtypes:
        start_time = time.time()
        costs = []
        reg_costs = []
        for i in range(args.i_start, len(source_clouds) if args.i_end==-1 else args.i_end):
            source_cloud = source_clouds[i]
            for target_cloud in target_clouds[i]:
                if args.visual_prior:
                    vis_cost_xy = ab_cost(source_cloud, target_cloud)
                else:
                    vis_cost_xy = None
                if regtype == 'rpm':
                    f, corr_nm = tps_rpm(source_cloud[:,:-3], target_cloud[:,:-3],
                                         vis_cost_xy = vis_cost_xy,
                                         plotting=args.plotting, plot_cb = plot_cb_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm") if args.output_folder else None,
                                                                                       args,
                                                                                       source_cloud[:,-3:],
                                                                                       target_cloud[:,-3:]))
                elif regtype == 'rpm-cheap':
                    f, corr_nm = tps_rpm(source_cloud[:,:-3], target_cloud[:,:-3],
                                         vis_cost_xy = vis_cost_xy, n_iter = N_ITER_CHEAP, em_iter = EM_ITER_CHEAP, 
                                         plotting=args.plotting, plot_cb = plot_cb_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm_cheap") if args.output_folder else None,
                                                                                       args,
                                                                                       source_cloud[:,-3:],
                                                                                       target_cloud[:,-3:]))
                elif regtype == 'rpm-bij':
                    x_nd = source_cloud[:,:3]
                    y_md = target_cloud[:,:3]
                    scaled_x_nd, _ = registration.unit_boxify(x_nd)
                    scaled_y_md, _ = registration.unit_boxify(y_md)
                    f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy,
                                                   plotting=args.plotting, plot_cb=plot_cb_bij_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm_bij") if args.output_folder else None,
                                                                                                   args,
                                                                                                   source_cloud[:,-3:],
                                                                                                   target_cloud[:,-3:]))
                elif regtype == 'rpm-bij-cheap':
                    x_nd = source_cloud[:,:3]
                    y_md = target_cloud[:,:3]
                    scaled_x_nd, _ = registration.unit_boxify(x_nd)
                    scaled_y_md, _ = registration.unit_boxify(y_md)
                    f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=10, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy, # Note registration_cost_cheap in rope_qlearn has a different rot_reg and outlierfrac
                                                   plotting=args.plotting, plot_cb=plot_cb_bij_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm_bij_cheap") if args.output_folder else None,
                                                                                                   args,
                                                                                                   source_cloud[:,-3:],
                                                                                                   target_cloud[:,-3:]))
                costs.append(f._cost)
                reg_costs.append(registration.tps_reg_cost(f))
        tps_costs.append(costs)
        tps_reg_costs.append(reg_costs)
        print regtype, "time elapsed", time.time() - start_time

    np.set_printoptions(suppress=True)
    
    print ""
    print "tps_costs"
    print args.regtypes
    print np.array(tps_costs).T
    
    print ""
    print "tps_reg_costs"
    print args.regtypes
    print np.array(tps_reg_costs).T

if __name__ == "__main__":
    main()
