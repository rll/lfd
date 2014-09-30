# Contains useful functions for evaluating on PR2 rope tying simulation.
# The purpose of this class is to eventually consolidate the various
# instantiations of do_task_eval.py
import argparse
from core import sim_util
import importlib, __builtin__
import util
import openravepy, trajoptpy
import h5py, numpy as np
from rapprentice import math_utils as mu
from string import lower

class EvalStats(object):
    def __init__(self, **kwargs):
        self.success = False
        self.feasible = False
        self.misgrasp = False
        self.generalized = False
        self.action_elapsed_time = 0
        self.exec_elapsed_time = 0
        for k in kwargs:
            setattr(self, k, kwargs[k])

def get_indexed_items(itemsfile, task_list=None, task_file=None, i_start=-1, i_end=-1):
    tasks = [] if task_list is None else task_list
    if task_file is not None:
        file = open(task_file, 'r')
        for line in file.xreadlines():
            try:
                tasks.append(int(line))
            except:
                print "get_specified_tasks:", line, "is not a valid task"
    if i_start != -1 or i_end != -1:
        if i_end == -1:
            i_end = len(itemsfile)
        if i_start == -1:
            i_start = 0
        tasks.extend(range(i_start, i_end))
    if not tasks:
        return sorted([item for item in itemsfile.iteritems() if item[0].isnumeric() or isinstance(item[0], int)], key=lambda item: int(item[0]))
    else:
        return [(unicode(t), itemsfile[unicode(t)]) for t in tasks]

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

def check_equal(g_or_d1, g_or_d2, parent_seq = None):
    if parent_seq == None:
        parent_seq = []
    keys1 = sorted(g_or_d1.keys())
    keys2 = sorted(g_or_d2.keys())
    if keys1 != keys2:
        return parent_seq
    cur_differences = []
    for k1, k2 in zip(keys1, keys2):
        if k1 != k2:
            cur_differences.append(parent_seq + [(k1, k2)])
        else:
            cur_seq = parent_seq + [k1]
            g1 = g_or_d1[k1]
            g2 = g_or_d2[k2]
            if isinstance(g1, h5py.Group) and isinstance(g2, h5py.Group):
                subtree_differences = check_equal(g1, g2, cur_seq)
                if subtree_differences:
                    cur_differences.extend(subtree_differences)
            elif np.any(g1[()] != g2[()]):
                cur_differences.append(parent_seq + [(k1, k2)])
    return sorted(cur_differences, key=lambda x: len(x))
    
    

def save_results_args(fname, args):
    # if args is already in the results file, make sure that the eval arguments are the same
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')

    if 'args' in result_file:
        loaded_args = group_or_dataset_to_obj(result_file['args'])
        if 'eval' not in vars(loaded_args):
            raise RuntimeError("The file doesn't have eval arguments")
        if 'eval' not in vars(args):
            raise RuntimeError("The current arguments doesn't have eval arguments")
        loaded_args_eval_dict = vars(loaded_args.eval)
        args_eval_dict = vars(args.eval)
        inconsistent_args = False
        inconsistent_args_msg = ""
        if set(loaded_args_eval_dict.keys()) != set(args_eval_dict.keys()):
            inconsistent_args = True
        for (k, args_eval_val) in args_eval_dict.iteritems():
            if inconsistent_args:
                break
            loaded_args_eval_val = loaded_args_eval_dict[k]
            if np.any(args_eval_val != loaded_args_eval_val):
                inconsistent_args = True
                inconsistent_args_msg = "%s, %s"%(loaded_args_eval_val, args_eval_val)
        if inconsistent_args:
            user_resp = raw_input("The arguments of the file and the current arguments have different eval arguments %s, overwrite?[y/N]"%inconsistent_args_msg)
            if lower(user_resp) == 'y':
                result_file.close()
                result_file = h5py.File(fname, 'w')
                add_obj_to_group(result_file, 'args', args)
            else:
                raise RuntimeError
    else:
        add_obj_to_group(result_file, 'args', args)
    result_file.close()

def load_results_args(fname):
    if fname is None:
        raise RuntimeError("Cannot load task results with an unspecified file name")
    result_file = h5py.File(fname, 'r')
    args = group_or_dataset_to_obj(result_file['args'])
    result_file.close()
    return args

def save_task_results_step(fname, task_index, step_index, results):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    if int(step_index) == 0:
        if task_index in result_file:
            del result_file[task_index]
        result_file.create_group(task_index)
    task_index = str(task_index)
    step_index = str(step_index)
    assert task_index in result_file, "Must call this function with step_index of 0 first"
    if step_index not in result_file[task_index]:
        step_group = result_file[task_index].create_group(step_index)
    else:
        step_group = result_file[task_index][step_index]
    add_obj_to_group(step_group, 'results', results)
    result_file.close()

def load_task_results_step(fname, task_index, step_index):
    if fname is None:
        raise RuntimeError("Cannot load task results with an unspecified file name")
    result_file = h5py.File(fname, 'r')
    task_index = str(task_index)
    step_index = str(step_index)
    step_group = result_file[task_index][step_index]
    results = group_or_dataset_to_obj(step_group['results'])
    result_file.close()
    return results

def traj_collisions(sim_env, full_traj, collision_dist_threshold, upsample=0):
    """
    Returns the set of collisions. 
    manip = Manipulator or list of indices
    """
    traj, dof_inds = full_traj
    sim_util.unwrap_in_place(traj, dof_inds=dof_inds)

    if upsample > 0:
        traj_up = mu.interp2d(np.linspace(0,1,upsample), np.linspace(0,1,len(traj)), traj)
    else:
        traj_up = traj
    cc = trajoptpy.GetCollisionChecker(sim_env.env)

    with openravepy.RobotStateSaver(sim_env.robot):
        sim_env.robot.SetActiveDOFs(dof_inds)
    
        col_times = []
        for (i,row) in enumerate(traj_up):
            sim_env.robot.SetActiveDOFValues(row)
            col_now = cc.BodyVsAll(sim_env.robot)
            #with util.suppress_stdout():
            #    col_now2 = cc.PlotCollisionGeometry()
            col_now = [cn for cn in col_now if cn.GetDistance() < collision_dist_threshold]
            if col_now:
                #print [cn.GetDistance() for cn in col_now]
                col_times.append(i)
                #print "trajopt.CollisionChecker: ", len(col_now)
            #print col_now2
        
    return col_times

def traj_is_safe(sim_env, full_traj, collision_dist_threshold, upsample=0):
    return traj_collisions(sim_env, full_traj, collision_dist_threshold, upsample) == []
