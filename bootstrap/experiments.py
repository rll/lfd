import argparse, h5py, do_task, sys, dhm_utils, shutil, os

import os.path as osp
import IPython as ipy
import numpy as np
import math
#import cloud
import cPickle as cp

from rapprentice import clouds
from rapprentice.colorize import colorize
from pdb import pm, set_trace
from file_utils import setup_bootstrap_file, create_bootstrap_item, gen_task_file, check_task_file

try:
    import do_task_floating
    reload(do_task_floating)
    from do_task_floating import sample_rope_state, TaskParameters, do_single_task
except:
    raise
    print "do_task_floating import failed, using do_task"
    from do_task import sample_rope_state
    from do_task import TaskParameters_floating as TaskParameters
    from do_task import do_single_task_floating as do_single_task


DS_SIZE = 0.03
DEFAULT_TREE_SIZES = [0, 30, 60, 90, 120]

def run_bootstrap(task_fname, action_fname, bootstrap_fname, burn_in = 40, tree_sizes = None, animate=False, no_cmat=False):
    """
    generates a bootstrapping tree
    taskfile has the training examples to use
    bootstrap_fname will be used as the file to create all of the bootstrapping trees
    tree_sizes controls the number of trees we want to build
    results for tree size i will be in bootstrap_fname_i.h5
    """
    if not tree_sizes:
        tree_sizes = DEFAULT_TREE_SIZES[:]
    taskf = h5py.File(task_fname, 'r')    
    assert len(taskf) >= burn_in + max(tree_sizes)
    taskf.close()
    task_ctr = 0
    setup_bootstrap_file(action_fname, bootstrap_fname)
    bootstrap_orig = osp.splitext(bootstrap_fname)[0] + '_orig.h5'
    shutil.copyfile(bootstrap_fname, bootstrap_orig)
    results = []
    for i in range(burn_in):
        print 'doing burn in {}/{}'.format(i, burn_in)
        res = run_example((task_fname, str(task_ctr), bootstrap_orig, bootstrap_fname, animate, no_cmat))
        results.append(res)
        task_ctr += 1                        
    for i in range(max(tree_sizes)):
        print 'doing bootstrapping {}/{}'.format(i, max(tree_sizes))
        if i in tree_sizes:
            bootstrap_i_fname = osp.splitext(bootstrap_fname)[0] + '_{}.h5'.format(i)
            shutil.copyfile(bootstrap_fname, bootstrap_i_fname)
        res = run_example((task_fname, str(task_ctr), bootstrap_fname, bootstrap_fname, animate, no_cmat))
        results.append(res)
        task_ctr += 1
    print 'success rate', sum(results)/float(len(results))
    return sum(results)/float(len(results))

def run_example((task_fname, task_id, action_fname, bootstrap_fname, animate, no_cmat)):
    """
    runs a knot-tie attempt for task_id (taken from taskfile
    possible actions are the expert demonstrations in actionfile
         assumed to be openable in 'r' mode
    if bootstrap_fname == '', then it won't save anything to that file
    set bootstrap_fname to add the results from the trial run to that file
         this is assumed to already be initialized
         this will append into that file and assumes that bootstrap_file has all the actions from actionfile in it         
    returns True if this is a knot-tie else returns False
    """
    taskfile = h5py.File(task_fname, 'r')
    init_xyz = taskfile[str(task_id)][:]
    taskfile.close()
    # currently set to test that correspondence trick does what we want
    task_params = TaskParameters(action_fname, init_xyz, animate=animate, no_cmat=no_cmat)
    task_results = do_single_task(task_params)
    if task_results['success'] and bootstrap_fname:
        try:
            bootf = h5py.File(bootstrap_fname, 'r+')
            for seg_values in task_results['seg_info']:
                cloud_xyz, parent, hmats, cmat = [seg_values[k] for k in ['cloud_xyz', 'parent', 'hmats', 'cmat']]
                children = []
                root_seg = str(bootf[str(parent)]['root_seg'][()])
                create_bootstrap_item(bootf, cloud_xyz, root_seg, str(parent), children, hmats, cmat)
        except:
            print 'encountered exception', sys.exc_info()
            print 'warning, bootstrap file may be malformed'
            raise
        finally:
            bootf.close()
    return task_results['success']


class CloudParams:
    def __init__(self):
        self.cmd_params      = None
        self.num_batches     = None
        self.start_batch_num = None
        self.end_batch_num   = None
        self.results_fname   = None
        self.env             = 'RSS3'
        self.vol             = 'iros_dat'
        self.core_type       = 'f2'

def create_test_params(local_task_fname, task_fname, action_fname):
    """
    The list of params returned by this is to be mapped to run_example
    """
    taskfile   = h5py.File(local_task_fname, 'r')
    ntasks     = len(taskfile.keys())
    taskfile.close()
    cmd_params = [(task_fname, i, action_fname, "", False) for i in xrange(ntasks)]
    return cmd_params


def run_tests_on_cloud(cloud_params, do_local=False):
    """
    make sure that task_fname and action_fname are available on the volume in the cloud.
    
    do_local : if true, the 'map' is done locally.
    """
    
    ntests     = len(cloud_params.cmd_params)
    batch_size = int(math.ceil(ntests/(cloud_params.num_batches+0.0)))

    batch_edges = batch_size*np.array(xrange(cloud_params.num_batches))[cloud_params.start_batch_num : cloud_params.end_batch_num]

    all_succ = []
    for i in xrange(len(batch_edges)):
        if i==len(batch_edges)-1:
            cmds = cloud_params.cmd_params[batch_edges[i]:]
        else:
            cmds = cloud_params.cmd_params[batch_edges[i]:min(batch_edges[i+1], ntests)]
        print colorize("calling on cloud : batch [%d/%d] "%(i, len(batch_edges)), "yellow", True)
        try:
            if not do_local:
                jids = cloud.map(run_example, cmds, _vol=cloud_params.vol, _env=cloud_params.env, _type=cloud_params.core_type)
                succ = cloud.result(jids)
                print colorize("\t got results for batch %d/%d "%(i, len(batch_edges)), "green", True)
            else:
                succ = map(run_example, cmds)
            all_succ += succ
        except Exception as e:
            print "Found exception %s. Not saving data for this demo."%e

    with open(cloud_params.results_fname, 'w') as f:
        print colorize("\t\t saved results in : %s"%cloud_params.results_fname, "green")
        cp.dump(all_succ, f)


def test_bootrun(bootrun_name='boot_1', do_nn=False, tree_sizes=[30,60,90,120], test_fname="eval_set.h5"):
    """
    @ res_dir       : the directory where the results from the test runs will be saved.
                      the saved results will be like: 
                         res_dir/<bootrun_name>_<tree_size>_res.cp
    @ bootrun_name  : the directory holding the actions from the bootstrap runs.
    @ do_nn         : the action file here is just the nearest neighbor file [use this for baseline]
    @ tree_sizes    : the sizes of bootstrap trees to run tests on.
    """
    cloud_bootstrapping_dir = "/home/picloud/sandbox/bootstrapping"
    data_dir           = osp.join(os.getenv("BOOTSTRAPPING_DIR"), "data")
    local_task_fname   = osp.join(data_dir, test_fname)
    task_fname         = osp.join(cloud_bootstrapping_dir, "data/%s"%test_fname) 

    if not do_nn:
        result_fnames      = [osp.join(data_dir, "test_results", "%s_%d_result.cp"%(bootrun_name, s)) for s in tree_sizes]
        test_action_fnames = [osp.join(cloud_bootstrapping_dir, "data", bootrun_name, 'test_bootstrapping_%d.h5'%s) for s in tree_sizes]
    else:
        test_basename      = osp.splitext(osp.basename(test_fname))[0]
        result_fnames      = [osp.join(data_dir, "test_results", "%s_%s_nn_result.cp"%(bootrun_name, test_basename))]
        test_action_fnames = [osp.join(cloud_bootstrapping_dir, "data", bootrun_name, 'test_bootstrapping_orig.h5')]


    for i in xrange(len(test_action_fnames)):
        cmd_params      = create_test_params(local_task_fname, task_fname, test_action_fnames[i])
        print colorize(" SUBMITTING %d jobs to run on the cloud"%len(cmd_params), "red", True)
        cloud_params    =  CloudParams()
        cloud_params.cmd_params      = cmd_params
        cloud_params.num_batches     = 1
        cloud_params.start_batch_num = 0
        cloud_params.end_batch_num   = 1 ## exclusive
        cloud_params.results_fname   = result_fnames[i]
        cloud_params.env             = 'RSS3'
        cloud_params.vol             = 'iros_dat'
        cloud_params.core_type       = 'f2'

        print colorize("running tests for file: %s"%test_action_fnames[i], "magenta", True)
        run_tests_on_cloud(cloud_params, False)


def run_example_test():
    boot_fname = 'data/test_bootstrapping.h5'
    try:
        os.remove(boot_fname)
    except:
        pass
    act_fname = 'data/actions.h5'
    task_fname = 'data/test_tasks.h5'
    return run_bootstrap(task_fname, act_fname, boot_fname, burn_in = 5, tree_sizes = [0])


def main():
    args = parse_arguments()
    boot_dir = args.bootstrapping_directory
    #TODO  Should the boot_fname be different?
    boot_fname = osp.join(boot_dir, 'test_bootstrapping.h5')
    try:
        os.remove(boot_fname)
    except:
        pass
    act_fname = args.actions_file
    task_fname = osp.join(boot_dir, 'tasks.h5')
    try:
        good_task_file = check_task_file(task_fname)
        if not good_task_file:
            raise
    except:
        gen_task_file(task_fname, 200, act_fname)
    if args.burn_in is not None:
        burn_in = args.burn_in
    else:
        burn_in = 40
    if args.tree_sizes:
        tree_sizes = args.tree_sizes
    else:
        tree_sizes = None
    success_rate = run_bootstrap(task_fname, act_fname, boot_fname, burn_in=burn_in, tree_sizes=tree_sizes, animate=args.animate, no_cmat=args.no_cmat)    
    import cPickle as cp
    res_fname = osp.join(boot_dir, 'res.cp')
    with open(res_fname, 'w') as f:
        cp.dump({'success_rate':success_rate, 'args':args}, f)
    return success_rate


def parse_arguments():
    import argparse

    usage = """
    Run {0} --help for a list of arguments
    Warning: This may modify existing hdf5 files.
    The task file should be in bootstrapping_directory/tasks.h5
    """.format(sys.argv[0])

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("actions_file", type=str,
                        help="The file that contains the original (probably human) demonstrations.")
    parser.add_argument("bootstrapping_directory", type=str,
                        help="The directory that contains or will contain the learned bootstrapped h5 files.")
    parser.add_argument("--animate", action='store_true', help='If included, then it will show the animation')
    parser.add_argument("--burn_in", type=int, default=None,
                        help="The number of burn-in iterations to run. The burn-in iterations only uses original segments")
    parser.add_argument("--tree_sizes", type=int, nargs="+", default=None,
                        help="A space separated list of the number of bootstrapping iterations each bootstrap file should be created from")
    parser.add_argument("--no_cmat", action='store_true')
    args = parser.parse_args()
    print "args =", args
    return args

def testing_main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_name", type=str,
                        help="name of the bootstrap directory like : boot_1, boot_2, ...")
    parser.add_argument("--baseline", action='store_true', help='If included, then only the original segments will be chosen [for baseline].')
    parser.add_argument("--tree_sizes", type=int, nargs="+", default=[30,60,90,120],
                        help="A space separated list of the number of bootstrapping iterations each bootstrap file should be created from")
    
    parser.add_argument("--test_fname", type=str, default="eval_set.h5",
                        help="name of test initial states file.")
    
    args = parser.parse_args()
    print args.tree_sizes
    test_bootrun(args.bootstrap_name, args.baseline, args.tree_sizes, test_fname=args.test_fname)


if __name__ == "__main__":
    main()
    #testing_main()


