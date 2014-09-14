import h5py
import numpy as np
import os.path as os
from rapprentice import clouds

def format_action_file(orig_fname, target_fname, args):
    demos = {}
    actions = h5py.File(orig_fname, 'r')
    for action, seg_info in actions.iteritems():
        if args.ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            x_nd = rope_nodes
            scene_state = GroundTruthRopeSceneState(rope_nodes, ROPE_RADIUS, upsample=args.eval.upsample, upsample_rad=args.eval.upsample_rad, downsample_size=args.eval.downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
            x_nd = full_cloud
            scene_state = SceneState(full_cloud, downsample_size=args.eval.downsample_size)
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
        bend_coefs = np.around(loglinspace(args.bend_coef_init, args.bend_coef_final, args.n_iter), 
                               BEND_COEF_DIGITS)

        solver_data = ds_and_precompute(x_nd, lr2ee_traj['l'][:, :3, 3], lr2ee_traj['r'][:, :3, 3], bend_coefs)

        demo = Demonstration(action, scene_state, aug_traj, solver_data = solver_data)
        
        demos[action] = demo
    actions.close()
    raw_file = h5py.File(target_fname, 'w')
    add_obj_to_group(raw_file, 'data', demos)
    raw_file.close()
    


def add_obj_to_group(group, k, v):
    if v is None:
        group[k] = 'None'
    elif (type(v) == dict or type(v) == list or type(v) == tuple) and len(v) == 0:
        group[k] = 'empty'
        group[k].attrs.create('value_type', type(v).__name__)
    elif type(v) == dict or type(v) == list or type(v) == tuple or hasattr(v, '__dict__'):
        vgroup = group.create_group(k)
        if type(v) == dict:
            d = v
        elif type(v) == list or type(v) == tuple:
            vgroup.attrs.create('value_type', type(v).__name__)
            d = dict((str(i),vi) for (i,vi) in enumerate(v))
        elif hasattr(v, '__dict__'):
            vgroup.attrs.create('value_type', type(v).__name__)
            vgroup.attrs.create('value_type_module', type(v).__module__)
            d = vars(v)
        for (vk,vv) in d.iteritems():
            add_obj_to_group(vgroup, vk, vv)
    else:
        group[k] = v
    return group

def group_or_dataset_to_obj(group_or_dataset):
    if 'value_type' in group_or_dataset.attrs.keys():
        if 'value_type_module' in group_or_dataset.attrs.keys():
            module = importlib.import_module(group_or_dataset.attrs['value_type_module'])
        else:
            module = __builtin__
        v_type = getattr(module, group_or_dataset.attrs['value_type'])
    else:
        v_type = None
    if isinstance(group_or_dataset, h5py.Group):
        group = group_or_dataset
        v_dict = {}
        for (gk,gv) in group.iteritems():
            v_dict[gk] = group_or_dataset_to_obj(gv)
        if v_type is not None:
            if v_type == tuple or v_type == list:
                v = v_type(zip(*sorted(v_dict.items(), key=lambda (vk, vv): int(vk)))[1])
            elif hasattr(v_type, '__dict__'):
                v = v_type.__new__(v_type)
                v.__dict__ = v_dict
        else:
            v = v_dict
    else:
        dataset = group_or_dataset
        if dataset[()] == 'None':
            v = None
        elif dataset[()] == 'empty':
            v = []
            if v_type is not None:
                v = v_type(v)
        else:
            v = dataset[()]
    return v

def gen_rot_sequence_task_file(taskfname, actionfname):
    """
    draw a sequence of training examples that consider rotating the pt clouds
    rotation is initially small and introduced so that we explore the manifold better
    generates 200 samples
    """
    min_theta = np.pi/8
    max_theta = np.pi
    burnin_length = 50
    num_samples = 200
    theta_vals = np.linspace(min_theta, max_theta, num_samples - burnin_length)
    min_rad, max_rad = 0.1, 0.1
    num_perturb_pts = 7
    taskfile = h5py.File(taskfname, 'w')
    actionfile = h5py.File(actionfname, 'r')
    try:
        for i in range(burnin_length):
            dhm_utils.one_l_print('Creating State {}/{}'.format(i, num_samples))
            with dhm_utils.suppress_stdout():
                taskfile[str(i)] = sample_rope_state(actionfile, perturb_points=num_perturb_pts,
                                                        min_rad=min_rad, max_rad=max_rad, rotation = min_theta)
        for j in range(burnin_length, num_samples):
            dhm_utils.one_l_print('Creating State {}/{}'.format(j, num_samples))
            print j, theta_vals[j-burnin_length]
            with dhm_utils.suppress_stdout():
                taskfile[str(j)] = sample_rope_state(actionfile, perturb_points=num_perturb_pts,
                                                        min_rad=min_rad, max_rad=max_rad, rotation = theta_vals[j-burnin_length])
        print ''
    except:
        print 'encountered exception', sys.exc_info()
        raise
    finally:                
        taskfile.close()
        actionfile.close()

def gen_task_file(taskfname, num_examples, actionfname, perturb_bounds=None, num_perturb_pts=7):
    """
    draw num_examples states from the initial state distribution defined by
    do_task.sample_rope_state
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
            dhm_utils.one_l_print('Creating State {}/{}'.format(i, num_examples))
            with dhm_utils.suppress_stdout():
                taskfile[str(i)] = sample_rope_state(actionfile, perturb_points=num_perturb_pts,
                                                        min_rad=min_rad, max_rad=max_rad, rotation=np.pi/4)
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

