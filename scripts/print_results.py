import argparse
import h5py

def estimate_performance(args, results_file):

    num_knots = 0
    num_tasks = len(results_file)-1

    start_i = min([int(i) for i in results_file.keys() if i != 'args'])
    succ = []
    failures = []
    goal_failures = []
    for i in range(start_i, num_tasks+start_i):
        num_steps = len(results_file[str(i)])
        found_goal = False
        is_knot = results_file[str(i)][str(num_steps-1)]['results']['knot'][()]
        if is_knot:
            num_knots += 1
            succ.append(i)
        else:
            failures.append(i)
        if args.report_goal:
            for j in range(num_steps):
                if results_file[str(i)][str(j)]['results']['found_goal'][()] and j<3:
                    found_goal = True
            if not(found_goal):
                goal_failures.append(i)
    print 'Failures:', failures
    print 'Successes', succ
    if args.report_goal:
        print 'Goal Failures:', goal_failures
    return num_knots, num_tasks, failures

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    parser.add_argument('--report_goal', action='store_true')

    args = parser.parse_args()
    
    results_file = h5py.File(args.results_file, 'r')
    
    num_successes, num_tasks, failures = estimate_performance(args, results_file)
    print "Successes / Total: %d/%d" % (num_successes, num_tasks)
    print "Success rate:", float(num_successes)/float(num_tasks)
