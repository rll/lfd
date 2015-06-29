#!/usr/bin/env python

import caffe, numpy as np
import IPython as ipy
from rapprentice import berkeley_pr2

def get_alexnet(net_prototxt, net_model, net_mean, small_cnn=False):
    # The three inputs are files that store the corresponding Alexnet info
    is_lenet = False
    if "lenet" == net_prototxt.split("_")[0]:
        is_lenet = True
        
    if is_lenet:
        net = caffe.Classifier(net_prototxt, net_model)
        net.set_phase_test()
        net.set_mode_gpu()
        net.set_input_scale('data', 1)
        net.set_channel_swap('data', (2,1,0))
    else:
        net = caffe.Classifier(net_prototxt, net_model)
        net.set_phase_test()
        net.set_mode_gpu()  # Can change to cpu instead of gpu
        if small_cnn:
            net.set_mean('data', np.load(net_mean)[0])
        else:
            net.set_mean('data', np.load(net_mean))
            #net.set_raw_scale('data', 255)
            #net.set_channel_swap('data', (2,1,0))

    return net

def cloud_backto_xy(xyz, T_w_k):
    cx = 320. - .5
    cy = 240. - .5

    xyz_k = (xyz - T_w_k[:3,3][None,:]).dot(T_w_k[:3, :3])
    x = xyz_k[:, 0] / xyz_k[:, 2] * berkeley_pr2.f + cx
    y = xyz_k[:, 1] / xyz_k[:, 2] * berkeley_pr2.f + cy
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    return np.concatenate((x, y), axis=1)

def predictLabel2D(xy, image, net):
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    patch_size = params[0][2]
    offset = np.round(patch_size / 2.).astype(int)
    
    patches_indices = []
    for i in range(len(xy)):
        x_start = xy[i, 0] - offset
        y_start = xy[i, 1] - offset
        
        patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
        #patch = patch / 255.0
        patch = patch.astype(float)
        #patch = np.asarray(patch[:, :, [2, 1, 0]])  # since the image is read using cv2
        patches_indices.append((patch, i))
        
    patches, indices = zip(*patches_indices)
    
    n_patch_images = len(patches)
    n_iterations = np.ceil(n_patch_images / np.double(n_parallel))
    n_iterations = int(n_iterations)
    
    rope_crossing_predicts = []
    for i in range(n_iterations):
        start_id = n_parallel * i
        end_id = min(n_parallel * (i + 1), n_patch_images)
        scores = net.predict(patches[start_id:end_id], oversample=True)
        if end_id == n_patch_images:
            scores = scores[:end_id-start_id, :] ### scores the the logistic regression result.
            
        # print scores
            
        predicts = np.argmax(scores, axis=1) # the label
        rope_crossing_predicts = np.concatenate((rope_crossing_predicts, predicts)).astype(int)
        
    return rope_crossing_predicts

def predictLabel3D(xyz, image, net, T_w_k=None):
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    patch_size = params[0][2]
    offset = np.round(patch_size / 2.).astype(int)

    if T_w_k is None:
        T_w_k = np.eye(4)
    height, width, channel = image.shape
    xy = np.round(cloud_backto_xy(xyz, T_w_k)).astype(int)
        
    valid_mask = (xy[:, 0] - offset >= 0) & (xy[:, 0] - offset + patch_size <= width) & (xy[:, 1] - offset >= 0) & (xy[:, 1] - offset + patch_size <= height)
    xy = xy[valid_mask]
    
    rope_crossing_predicts_valid_points = predictCrossing2D(xy, image, net)
            
    rope_crossing_predicts = np.empty((len(xyz), 1))
    rope_crossing_predicts.fill(-1)
    rope_crossing_predicts[valid_mask] = np.expand_dims(rope_crossing_predicts_valid_points, axis=1)

    return rope_crossing_predicts

    print 
def predictCrossing2D(xy, image, net, image_name = None):
    # Same as predictLabel2D, but returns scores and features as well
    params = [(k, v.data.shape) for k, v in net.blobs.items()]
    n_parallel = params[0][1][0]
    patch_size = params[0][1][2]
    offset = np.round(patch_size / 2.).astype(int)

    patches_indices = []
    for i in range(len(xy)):
        x_start = xy[i, 0] - offset
        y_start = xy[i, 1] - offset
        patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
        #patch = patch / 255.0
        patch = patch.astype(float)
        #patch = np.asarray(patch[:, :, [2, 1, 0]])
        patches_indices.append((patch, i))

    patches, indices = zip(*patches_indices)
    n_patch_images = len(patches)
    n_iterations = np.ceil(n_patch_images / np.double(n_parallel))
    n_iterations = int(n_iterations)
    rope_crossing_predicts = []
    rope_crossing_scores = None

    rope_features = {}
    for param in params:
        if "ip" in param[0] or "fc" in param[0] or "pool" in param[0]:
            rope_features[param[0]] = np.zeros([0, param[1][1], param[1][2], param[1][3]])

    for i in range(n_iterations):
        start_id = n_parallel * i
        end_id = min(n_parallel * (i + 1), n_patch_images)
        scores = net.predict(patches[start_id:end_id], oversample=True)
        if end_id == n_patch_images:
            scores = scores[:end_id-start_id, :]
        for feature_name in rope_features.keys():
            feat = net.blobs[feature_name].data.copy()
            if end_id == n_patch_images:
                feat = feat[:end_id-start_id, :, :, :]
            rope_features[feature_name] = np.concatenate([rope_features[feature_name], feat], axis=0)

        # print scores
        num_points, num_classes = scores.shape
        if rope_crossing_scores is None:
            rope_crossing_scores = np.zeros([0, num_classes])

        rope_crossing_scores = np.concatenate([rope_crossing_scores, scores], axis=0)
        predicts = np.argmax(scores, axis=1)
        rope_crossing_predicts = np.concatenate((rope_crossing_predicts, predicts)).astype(int)

    return rope_crossing_predicts, rope_crossing_scores, rope_features

def predictCrossing3D(xyz, image, net, T_w_k=None, image_name=None):
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    patch_size = params[0][2]
    offset = np.round(patch_size / 2.).astype(int)

    if T_w_k is None:
        T_w_k = np.eye(4)
    height, width, channel = image.shape
    xy = np.round(cloud_backto_xy(xyz, T_w_k)).astype(int)
    valid_mask = (xy[:, 0] - offset >= 0) & (xy[:, 0] - offset + patch_size <= width) & (xy[:, 1] - offset >= 0) & (xy[:, 1] - offset + patch_size <= height)
    xy = xy[valid_mask]

    rope_crossing_predicts_valid_points, rope_crossing_scores_valid_points, rope_features_valid_points = predictCrossing2D(xy, image, net, image_name)
    return rope_crossing_predicts_valid_points, rope_crossing_scores_valid_points, rope_features_valid_points, valid_mask
