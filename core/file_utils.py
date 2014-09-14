import h5py
import numpy as np
import os.path as osp
from rapprentice import clouds
from demonstration import Demonstration, AugmentedTrajectory, SceneState, GroundTruthRopeSceneState
from constants import DEFAULT_LAMBDA, N_ITER_CHEAP, DS_SIZE, BEND_COEF_DIGITS, ROPE_RADIUS
from registration import loglinspace
from tpsopt.file_utils import *
import sim_util
import sys


def format_action_file(actions, raw_file=None, ground_truth=False, upsample=0, upsample_rad=1, downsample_size=DS_SIZE,
                       bend_limits = DEFAULT_LAMBDA, n_iter=N_ITER_CHEAP):
    if type(actions) is str:
        if raw_file is None:
            if ground_truth:
                raw_file = osp.splitext(actions)[0] + '.processed_ground_truth.h5'
            else:
                raw_file = osp.splitext(actions)[0] + '.processed.h5'
        # so that we safely close files
        actions = h5py.File(actions, 'r')
        raw_file = h5py.File(raw_file, 'w')
        try:            
            format_action_file(actions, raw_file, ground_truth=ground_truth, upsample=upsample, 
                               upsample_rad=upsample_rad, downsample_size=downsample_size,
                               bend_limits = bend_limits, n_iter=n_iter)
        finally:
            actions.close()
            raw_file.close()
        return
    (bend_coef_init, bend_coef_final) = bend_limits
    n_actions = len(actions)
    for i, (action, seg_info) in enumerate(actions.iteritems()):
        sim_util.one_l_print('formated {}/{} demonstrations'.format(i, n_actions))
        if ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            x_nd = rope_nodes
            scene_state = GroundTruthRopeSceneState(rope_nodes, ROPE_RADIUS, upsample=upsample, upsample_rad=upsample_rad, downsample_size=downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
            x_nd = full_cloud
            scene_state = SceneState(full_cloud, downsample_size=downsample_size)
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
        aug_traj = AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        bend_coefs = np.around(loglinspace(bend_coef_init, bend_coef_final, n_iter), 
                               BEND_COEF_DIGITS)

        demo = Demonstration(action, scene_state, aug_traj)
        demo.compute_solver_data(bend_coefs)
        add_obj_to_group(raw_file, action, demo)



def gen_task_file(taskfname, num_examples, actionfname, perturb_bounds=None, num_perturb_pts=7, rotation=0):
    """
    draw num_examples states from the initial state distribution defined by sim_util.sample_rope_state
    using intial states from actionfile

    writes results to fname
    """
    if not perturb_bounds:
        min_rad, max_rad = 0.1, 0.1
    else:
        min_rad, max_rad = perturb_bounds
    taskfile = h5py.File(taskfname, 'w')
    actionfile = h5py.File(actionfname, 'r')
    try:
        for i in range(num_examples):
            sim_util.one_l_print('Creating State {}/{}'.format(i, num_examples))
            with sim_util.suppress_stdout():
                rope_nodes, demo_id = sim_util.sample_rope_state(actionfile, perturb_points=num_perturb_pts,
                                                         min_rad=min_rad, max_rad=max_rad, rotation=rotation)
                new_g = taskfile.create_group(str(i))
                new_g['rope_nodes'] = rope_nodes
                new_g['demo_id'] = demo_id
        print ''
    except:
        print 'encountered exception', sys.exc_info()
        raise
    finally:                
        taskfile.close()
        actionfile.close()
    assert check_task_file(taskfname)

def check_task_file(fname):
    """
    probably unecessary, but checks that a task file is properly labelled sequentially
    """
    f = h5py.File(fname, 'r')
    success = True
    for i in range(len(f)):
        if str(i) not in f:
            print 'task file {} is missing key {}'.format(fname, i)
            success = False
    f.close()
    return success

