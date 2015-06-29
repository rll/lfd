import numpy as np, random, scipy, math
import scipy.spatial.distance as ssd
from scipy import spatial
from scipy.stats import entropy                                             

from rapprentice import clouds
from lfd.rapprentice import tps_registration

from alexnet_helper import predictCrossing3D

NUM_CORNER_PTS_TO_ADD = 100
CORNER_LABELS = set([1,2,3])  # Set these to your Alexnet corner labels
BACKGROUND_LABEL = 0  # Set this to the value of the background label

def ab_cost_with_threshold(xyzrgb1, xyzrgb2, threshold):
    lab1 = tps_registration.rgb2lab(xyzrgb1[:,-3:])
    lab2 = tps_registration.rgb2lab(xyzrgb2[:,-3:])
    cost = ssd.cdist(lab1[:,1:], lab2[:,1:], 'euclidean')
    # Normalize costs to be from 0 to 1; round down
    # to zero if the cost is below a certain threshold, and round
    # up to 1 otherwise
    cost = cost / float(np.max(cost)) > threshold
    return cost

def bgr_to_rgb(bgr_cloud):                                                      
    # Change range to 0-1 by dividing by 255                                    
    # Reverse order from BGR to RGB
    if np.max(bgr_cloud) > 1:
        bgr_cloud = bgr_cloud / 255.0

    n,d = bgr_cloud.shape                                                       
    rgb_cloud = np.zeros((n,d))
    rgb_cloud[:,0] = bgr_cloud[:,2]
    rgb_cloud[:,1] = bgr_cloud[:,1]
    rgb_cloud[:,2] = bgr_cloud[:,0]
    return rgb_cloud

def chi_squared_hist_distance(hist1, hist2):
    dist = 0
    for i in xrange(len(hist1)):
        avg = (hist1[i] + hist2[i]) / 2
        if avg == 0:
            continue
        dist += (hist1[i] - hist2[i])**2 / avg
    return dist

def get_labels_distance(labels1, labels2):
    max_value = max(max(labels1), max(labels2))
    labels1_hist = [0]*(max_value+1)
    labels2_hist = [0]*(max_value+1)
    labels1_list = list(labels1)
    labels2_list = list(labels2)
    for i in xrange(0,len(labels1_hist)):
        if i == BACKGROUND_LABEL:
            continue
        labels1_hist[i] = labels1_list.count(i) / float(len(labels1_list))
        labels2_hist[i] = labels2_list.count(i) / float(len(labels2_list))
    labels1_hist = labels1_hist / np.sum(np.array(labels1_hist))
    labels2_hist = labels2_hist / np.sum(np.array(labels2_hist))

    # Can compare ordering of the two lists
    #sorted_indices1 = [i[0] for i in sorted(enumerate(labels1_hist), key=lambda x:x[1])]
    #sorted_indices2 = [i[0] for i in sorted(enumerate(labels2_hist), key=lambda x:x[1])]

    #tau = scipy.stats.kendalltau(scipy.asarray(sorted_indices1), scipy.asarray(sorted_indices2))[0]
    #return (1 - tau)  # Doesn't work -- too many have same ordering
    
    #print "Sorted indices:", sorted_indices1
    #print "Sorted indices:", sorted_indices2
    #print "Should be normalized:", labels1_hist
    #print "Should be normalized:", labels2_hist

    #return np.linalg.norm(np.array(labels1_hist) - np.array(labels2_hist))
    return chi_squared_hist_distance(labels1_hist, labels2_hist)

def get_label_pairs_distance(labels1, labels2):
    # Returns distance between each pair of label types, in the order:
    #     [dist_0_0, dist_0_1, dist_0_2, ..., dist_10_9, dist_10_10]
    max_value = max(max(labels1), max(labels2))
    labels1_hist = [0]*(max_value+1)
    labels2_hist = [0]*(max_value+1)
    labels1_list = list(labels1)
    labels2_list = list(labels2)
    for i in xrange(len(labels1_hist)):
        if i == BACKGROUND_LABEL:
            continue
        labels1_hist[i] = labels1_list.count(i) / float(len(labels1_list))
        labels2_hist[i] = labels2_list.count(i) / float(len(labels2_list))
    pairs_dist = np.zeros((len(labels1_hist), len(labels2_hist)))
    for i in xrange(len(labels1_hist)):
        for j in xrange(len(labels2_hist)):
            pairs_dist[i][j] = labels1_hist[i] - labels2_hist[j]
    return tuple(pairs_dist.reshape((len(labels1_hist)*len(labels2_hist),)).tolist())

def get_xyzrgb_downsampled_cloud(cloud_xyzrgb, ds_size):
    n,d = cloud_xyzrgb.shape
    assert d == 6, "XYZRGB downsampling requires cloud with dimension 6"
    downsampled_xyz = clouds.downsample(cloud_xyzrgb[:,:-3], ds_size)

    # To each downsampled point, assign color of closest original point
    tree = spatial.KDTree(cloud_xyzrgb[:,:-3])
    [_, i] = tree.query(downsampled_xyz)
    n,d = downsampled_xyz.shape
    downsampled_xyzrgb = np.zeros((n,d+3))
    downsampled_xyzrgb[:,:d] = downsampled_xyz
    cloud_xyzrgb = cloud_xyzrgb[()]
    downsampled_xyzrgb[:,d:] = cloud_xyzrgb[i, d:]
    return downsampled_xyzrgb

def get_potential_corner_pts(xyz_ds, xyz_full, alexnet_features, rgb, net, T_w_k):
    # xyz_ds and xyz_full should already be in RGB, not BGR
    label_predicts_ds = alexnet_features[0]
    valid_mask_ds = alexnet_features[3]
    xyz_ds = xyz_ds[valid_mask_ds,:]
    corner_pts = np.array([xyz_ds[i,:] for i in range(len(xyz_ds)) if label_predicts_ds[i] in CORNER_LABELS])
    if len(corner_pts) == 0:
        print "There are no corner points"
        return (None, [], [], [], [])
    tree = spatial.KDTree(xyz_full[:,:-3])
    indices = tree.query_ball_point(corner_pts[:,:-3], 0.01)
    indices_nodups = set()
    for index_list in indices:
        indices_nodups.update(index_list)
    indices_nodups = random.sample(indices_nodups, min(len(indices_nodups), NUM_CORNER_PTS_TO_ADD)) 
    print "\tNumber of potential corner pts:", len(indices_nodups)
    pts_to_add = xyz_full[list(indices_nodups),:]

    label_predicts, label_scores, label_features, valid_mask = \
            predictCrossing3D(pts_to_add[:,:3], rgb, net, T_w_k=T_w_k)
    print "\tNumber of actual corner pts:", len([1 for p in label_predicts if p in CORNER_LABELS])
    return (pts_to_add, label_predicts, label_scores, label_features, valid_mask)

def get_alexnet_features(xyz_full_orig, rgb, T_w_k, net, ds_size, args, use_vis=False):
    # xyz_full should still be in BGR, not RGB
    # adds corners if args.extra_corners == 1
    xyz_full = np.copy(xyz_full_orig)
    xyz_full[:,-3:] = bgr_to_rgb(xyz_full[:,-3:])

    if args.downsample:
        xyz_ds = get_xyzrgb_downsampled_cloud(xyz_full, ds_size)
    else:
        xyz_ds = xyz_full

    if not use_vis and not args.use_vis:
        return (xyz_ds, None, None, None)

    assert net != None, "If using visual features, must provide a trained net"
    alexnet_features = predictCrossing3D(xyz_ds[:,:3], rgb, net, T_w_k=T_w_k)
    if not args.extra_corners or not args.use_vis:
        return (xyz_ds, None, alexnet_features, None)

    potential_corner_pts, label_predicts_add, label_scores_add, label_features_add, valid_mask_add = \
        get_potential_corner_pts(xyz_ds, xyz_full, alexnet_features, rgb, net, T_w_k)
    if potential_corner_pts == None:
        return (xyz_ds, xyz_ds, alexnet_features, alexnet_features)
    xyz_ds_100corners = np.concatenate((xyz_ds, potential_corner_pts), axis=0)
    label_predicts0 = np.concatenate((alexnet_features[0], label_predicts_add), axis=0)
    label_scores0 = np.concatenate((alexnet_features[1], label_scores_add), axis=0)
    label_features0 = {}
    for f_key in alexnet_features[2]:
        label_features0[f_key] = np.concatenate((alexnet_features[2][f_key], label_features_add[f_key]), axis=0)
    valid_mask0 = np.concatenate((alexnet_features[3], valid_mask_add), axis=0)
    alexnet_features_100corners = [label_predicts0, label_scores0, label_features0, valid_mask0]

    return (xyz_ds, xyz_ds_100corners, alexnet_features, alexnet_features_100corners)

def kl_divergence(scores1, scores2):
    entropy_1to2 = entropy(scipy.asarray(scores1), scipy.asarray(scores2))
    entropy_2to1 = entropy(scipy.asarray(scores2), scipy.asarray(scores1))
    if math.isinf(entropy_1to2):
        return entropy_2to1
    if math.isinf(entropy_2to1):
        return entropy_1to2

    return  (entropy_1to2 + entropy_2to1) / 2.0

def compute_score_costs(all_scores1, all_scores2):
    n_scores1 = len(all_scores1)                                                    
    n_scores2 = len(all_scores2)                                                    
    score_cost_matrix = np.zeros([n_scores1, n_scores2])
    for i in range(n_scores1):                                                  
        for j in range(n_scores2):                                              
            # To make the distance symmetric, make it (D_{KL}(P||Q) + D_{KL}(Q||P)) / 2
            score_cost_matrix[i, j] = kl_divergence(all_scores1[i], all_scores2[j])
                                                                                
    return score_cost_matrix

def vis_cost_fn(demo_state, test_state, beta = 1):
    # Returns cost matrix, to be multipled directly into the prior on point correspondences
    # Should already be exponentiated and (optionally) normalized so the largest value is one
    
    (label_predicts0, label_scores0, label_features0, valid_mask0) = demo_state.alexnet_features
    (label_predicts1, label_scores1, label_features1, valid_mask1) = test_state.alexnet_features
    old_xyz_labels = label_predicts0
    new_xyz_labels = label_predicts1

    # If there are extra points, label_predicts{0,1} will be the number of points in the original
    # point clouds, and demo_state.cloud and test_state.cloud include the extra points
    orig_num_pts = len(valid_mask0) + len(valid_mask1)
    old_xyz = demo_state.cloud[valid_mask0,:]
    new_xyz = test_state.cloud[valid_mask1,:]
    print (orig_num_pts - len(old_xyz) - len(new_xyz)), "points ignored because out of image range"

    vis_cost_xy = beta * compute_score_costs(label_scores0, label_scores1)
    #vis_cost_xy = BETA * ab_cost_with_threshold(old_xyz, new_xyz, THRESHOLD) # for color

    assert len(demo_state.cloud) - len(valid_mask0) == len(test_state.cloud) - len(valid_mask1), \
           "Must have the same number of extra points for demo and test states"

    if len(demo_state.cloud) - len(valid_mask0) > 0:
        # Update visual cost matrix to account for extra points (that were added in earlier)
        print "Number of extra points: ", len(demo_state.cloud) - len(valid_mask0)
        num_source_pts, num_target_pts = vis_cost_xy.shape
        complete_vis_cost_xy = np.max(vis_cost_xy) * np.ones((len(demo_state.get_valid_xyzrgb_cloud()), len(test_state.get_valid_xyzrgb_cloud())))
        (num_source_extra_pts, num_target_extra_pts) = complete_vis_cost_xy.shape
        for i in xrange(len(demo_state.cloud) - len(label_predicts0)):
            complete_vis_cost_xy[num_source_extra_pts-(i+1)][num_target_extra_pts-(i+1)] = 0
        complete_vis_cost_xy[0:num_source_pts, 0:num_target_pts] = vis_cost_xy
        vis_cost_xy = complete_vis_cost_xy

    prior = np.exp(-vis_cost_xy)
    prior /= (prior.max())
    return (vis_cost_xy, prior)

def new_vis_cost_fn(demo_state, test_state, beta = 1, corners_mult = 1):
    # Returns cost matrix, to be multipled directly into the prior on point correspondences
    # Should already be exponentiated and (optionally) normalized so the largest value is one
    
    (label_predicts0, label_scores0, label_features0, valid_mask0) = demo_state.alexnet_features
    (label_predicts1, label_scores1, label_features1, valid_mask1) = test_state.alexnet_features
    old_xyz_labels = label_predicts0
    new_xyz_labels = label_predicts1

    # If there are extra points, label_predicts{0,1} will be the number of points in the original
    # point clouds, and demo_state.cloud and test_state.cloud include the extra points
    orig_num_pts = len(valid_mask0) + len(valid_mask1)
    old_xyz = demo_state.cloud[valid_mask0,:]
    new_xyz = test_state.cloud[valid_mask1,:]
    print (orig_num_pts - len(old_xyz) - len(new_xyz)), "points ignored because out of image range"

    #vis_cost_xy = compute_score_costs(label_scores0, label_scores1)
    vis_cost_xy = np.zeros([len(label_predicts0), len(label_predicts1)])
    for i in range(len(label_predicts0)):
        for j in range(len(label_predicts1)):
            if label_predicts0[i] in CORNER_LABELS and label_predicts1[j] not in CORNER_LABELS:
                #vis_cost_xy[i,j] *= corners_mult
                vis_cost_xy[i,j] = corners_mult
            if label_predicts0[i] not in CORNER_LABELS and label_predicts1[j] in CORNER_LABELS:
                #vis_cost_xy[i,j] *= corners_mult
                vis_cost_xy[i,j] = corners_mult

    vis_cost_xy = beta * vis_cost_xy
    #vis_cost_xy = BETA * ab_cost_with_threshold(old_xyz, new_xyz, THRESHOLD) # for color

    assert len(demo_state.cloud) - len(valid_mask0) == len(test_state.cloud) - len(valid_mask1), \
           "Must have the same number of extra points for demo and test states"

    if len(demo_state.cloud) - len(valid_mask0) > 0:
        # Update visual cost matrix to account for extra points (that were added in earlier)
        print "Number of extra points: ", len(demo_state.cloud) - len(valid_mask0)
        num_source_pts, num_target_pts = vis_cost_xy.shape
        complete_vis_cost_xy = np.max(vis_cost_xy) * np.ones((len(demo_state.get_valid_xyzrgb_cloud()), len(test_state.get_valid_xyzrgb_cloud())))
        (num_source_extra_pts, num_target_extra_pts) = complete_vis_cost_xy.shape
        for i in xrange(len(demo_state.cloud) - len(label_predicts0)):
            complete_vis_cost_xy[num_source_extra_pts-(i+1)][num_target_extra_pts-(i+1)] = 0
        complete_vis_cost_xy[0:num_source_pts, 0:num_target_pts] = vis_cost_xy
        vis_cost_xy = complete_vis_cost_xy

    prior = np.exp(-vis_cost_xy)
    prior /= (prior.max())
    return (vis_cost_xy, prior)
