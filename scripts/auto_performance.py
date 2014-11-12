import argparse
import h5py
from lfd.rapprentice import eval_util

def estimate_performance(fname):
    results_file = h5py.File(fname, 'r')
    loadresult_items = eval_util.get_indexed_items(results_file)

    num_knots = 0
    num_misgrasps = 0
    num_infeasible = 0
    action_time = 0
    exec_time = 0

    for i_task, task_info in loadresult_items:
        knot_exists = False
        infeasible = False
        misgrasp = False

        for i_step in range(len(task_info)):
            results = eval_util.load_task_results_step(fname, i_task, i_step)

            eval_stats = results['eval_stats']
            misgrasp |= eval_stats.misgrasp
            infeasible |= not eval_stats.feasible
            action_time += eval_stats.action_elapsed_time
            exec_time += eval_stats.exec_elapsed_time
            
            if results['knot']:
                knot_exists = True
            elif i_step == len(task_info)-1:
                print i_task
        
        if infeasible:
            num_infeasible += 1
        if misgrasp:
            num_misgrasps += 1

        if knot_exists:
            num_knots += 1
    num_tasks = len(loadresult_items)
    
    print "# Misgrasps:", num_misgrasps
    print "# Infeasible:", num_infeasible
    print "Time taken to choose demo:", action_time, "seconds"
    print "Time taken to warp and execute demo:", exec_time, "seconds"
    return num_knots, num_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()
    
    results_file = h5py.File(args.results_file, 'r')
    
    num_successes, num_tasks = estimate_performance(args.results_file)
    print "Successes / Total: %d/%d" % (num_successes, num_tasks)
    print "Success rate:", float(num_successes)/float(num_tasks)
