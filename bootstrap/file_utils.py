import h5py
import numpy as np
import os.path as os
from rapprentice import clouds

DS_SIZE = 0.025

def setup_bootstrap_file(action_fname, bootstrap_fname):
    """
    copies over the relevant fields of action file to a bootstrap_file so that we can use the 
    resulting file for run_example
    """
    print action_fname, bootstrap_fname
    actfile = h5py.File(action_fname, 'r')
    bootfile = h5py.File(bootstrap_fname, 'w')
    for seg_name, seg_info in actfile.iteritems():
        seg_name = str(seg_name)
        cloud_xyz = clouds.downsample(seg_info['cloud_xyz'][:], DS_SIZE)
        # cloud_xyz = seg_info['cloud_xyz'][:]
        hmats = dict((lr, seg_info['{}_gripper_tool_frame'.format(lr)]['hmat'][:]) for lr in 'lr')
        cmat = np.eye(cloud_xyz.shape[0])
        gripper_joints = dict(('{}_gripper_joint'.format(lr), seg_info['{}_gripper_joint'.format(lr)][:]) for lr in 'lr')
        create_bootstrap_item(outfile=bootfile, cloud_xyz=cloud_xyz, root_seg=seg_name, parent=seg_name,
                              children=[], hmats=hmats, cmat=cmat, other_items=gripper_joints,
                              update_parent=False, seg_name=seg_name)
    actfile.close()
    assert check_bootstrap_file(bootstrap_fname, action_fname)
    return bootfile


def create_bootstrap_item(outfile, cloud_xyz, root_seg, parent, children, hmats, 
                          cmat, other_items = None, update_parent=True, seg_name=None):
    if not seg_name:
        seg_name = str(len(outfile))
    assert seg_name not in outfile, 'created duplicate segment in bootstrap file'

    g = outfile.create_group(seg_name)
    g['cloud_xyz'] = cloud_xyz # pt cloud associated with this action
    g['root_seg'] = root_seg # string that points to the root segment associated with this segment
    g['parent'] = parent     # string that points to the parent that this segment was matched to
    g['children'] = children if children else 0 # will contain a list of pointers to children for this node
                                                # initialized to be 0 b/c you can't store a 0-sized list in h5
    hmat_g = g.create_group('hmats')
    for lr in 'lr':
        hmat_g[lr] = hmats[lr] # list of hmats that contains the trajectory for the lr gripper in this segment    

    root_xyz = outfile[root_seg]['cloud_xyz'][:]
    root_n = root_xyz.shape[0]
    seg_m = cloud_xyz.shape[0]
    # do this check here, b/c we don't have to explicitly check if root_seg == seg_name
    assert cmat.shape == (root_n, seg_m), 'correspondence matrix formatted wrong'
    g['cmat'] = cmat

    if update_parent:
        parent_children = outfile[parent]['children'][()]
        del outfile[parent]['children']
        if parent_children == 0:
            parent_children = []
        parent_children = np.append(parent_children, [seg_name])
        outfile[parent]['children'] = parent_children
    if other_items:
        for k, v in other_items.iteritems():
            g[k] = v
    outfile.flush()
    return seg_name

def check_bootstrap_file(bootstrap_fname, orig_fname):
    """
    checks that bootstrap file is properly formatted
    assumes bootstrap_file was generated from orig_file
    this will return False is the file is improperly formatted
        - all the actions in orig_file are in bootstrap_file as their own parent
        - all top level entries have the correct fields
        - all parent and child pointers exist and point to each other
    returns True if file is formatted correctly
    """
    required_keys = ['children', 'cloud_xyz', 'cmat', 'hmats', 'parent', 'root_seg']
    bootf = h5py.File(bootstrap_fname, 'r')
    origf = h5py.File(orig_fname, 'r')
    success = True
    try:
        for seg_name in origf: # do original action checks
            seg_name = str(seg_name)
            if seg_name not in bootf:
                print 'original action {} from {} not in {}'.format(seg_name, orig_fname, bootstrap_fname)
                success = False
            for lr in 'lr':
                if '{}_gripper_joint'.format(lr) not in bootf[seg_name]:
                    print 'boostrap file {} root segment {} missing {}_gripper_joint'.format(bootstrap_fname, seg_name, lr)
                    success = False
        for seg_name, seg_info in bootf.iteritems():
            seg_name = str(seg_name)
            for k in required_keys:
                if k not in seg_info:
                    print 'bootstrap file {} segment {} missing key {}'.format(bootstrap_fname, seg_name, k)
            for lr in 'lr':
                if lr not in seg_info['hmats']:
                    print 'boostrap file {} segment {} missing {} hmats'.format(bootstrap_fname, seg_name, lr)
                    success = False
            parent = str(seg_info['parent'][()])
            if parent not in bootf:
                print 'boostrap file {} missing parent {} for segment {}'.format(bootstrap_fname, parent, seg_name)
                success = False
            parent_children = bootf[parent]['children'][()]
            if parent != seg_name and seg_name not in parent_children:
                print 'boostrap file {} parent {} does not have pointer to child {}'.format(bootstrap_fname, parent, seg_name)
                success = False
            root_seg = str(seg_info['root_seg'][()])
            if root_seg not in bootf:
                print 'boostrap file {} missing root_seg {} for segment {}'.format(bootstrap_fname, root_seg, seg_name)
                success = False
            root_n = bootf[root_seg]['cloud_xyz'][:].shape[0]
            seg_m = seg_info['cloud_xyz'][:].shape[0]
            if seg_info['cmat'][:].shape != (root_n, seg_m):
                print 'boostrap file {} cmat for segment {} has wrong dimension'.format(bootstrap_fname, root_seg, seg_name)
                print 'is', seg_info['cloud_xyz'][:].shape[0], 'should be', (root_n, seg_m)
                success = False
    except:
        print 'encountered exception', sys.exc_info()
        bootf.close()
        origf.close()
        success = False
        raise
    return success

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

