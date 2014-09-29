import argparse
import h5py

def estimate_performance(results_file):

    num_knots = 0
    num_tasks = len(results_file)-1
    for i in range(num_tasks):
        num_steps = len(results_file[str(i)])

        is_knot = results_file[str(i)][str(num_steps-1)]['results']['knot'][()]
        if is_knot:
            num_knots += 1
    return num_knots, num_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()
    
    results_file = h5py.File(args.results_file, 'r')
    
    num_successes, num_tasks = estimate_performance(results_file)
    print "Successes / Total: %d/%d" % (num_successes, num_tasks)
    print "Success rate:", float(num_successes)/float(num_tasks)
