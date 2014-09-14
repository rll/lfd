import scripts.eval
from scripts.eval import build_parser, setup_log_file, set_global_vars, setup_lfd_environment_sim, eval_on_holdout_parallel, \
    eval_on_holdout, setup_registration_and_trajectory_transferer
from core.action_selection import GreedyActionSelection
from core.file_utils import fname_to_obj
from rapprentice import eval_util

import trajoptpy
import time

def parse_input_args():
    parser = build_parser()
    parser.add_argument("--train_sizes", type=int, nargs="+", default=None,
                        help="A space separated list of the number of bootstrapping iterations each bootstrap file should be created from")
    args = parser.parse_args()
    args.eval.ground_truth = True
    args.eval.gpu = True
    return args 

def main():
    args = parse_input_args()

    if args.subparser_name == "eval":
        eval_util.save_results_args(args.resultfile, args)
    elif args.subparser_name == "replay":
        loaded_args = eval_util.load_results_args(args.replay.loadresultfile)
        assert 'eval' not in vars(args)
        args.eval = loaded_args.eval
    else:
        raise RuntimeError("Invalid subparser name")
    
    setup_log_file(args)
    print 'reading action data'
    bootstrap_data = fname_to_obj(args.eval.actionfile)
    print 'max library size:\t{}'.format(max(len(x) for x in bootstrap_data.values()))
    scripts.eval.GlobalVars.demos = bootstrap_data.values()[0]
    trajoptpy.SetInteractive(args.interactive)
    print 'building sim env'
    lfd_env, sim = setup_lfd_environment_sim(args)
    results = {}
    if args.subparser_name == "eval":
        start = time.time()
        if args.eval.parallel:
            eval_on_holdout_parallel(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
        else:
            for s in args.train_sizes:
                scripts.eval.GlobalVars.demos = bootstrap_data[str(s)]
                reg_and_traj_transferer = setup_registration_and_trajectory_transferer(args, sim)
                action_selection = GreedyActionSelection(reg_and_traj_transferer.registration_factory)
                results[s] = eval_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
                print "eval time is:\t{}".format(time.time() - start)
            print "Eval Results: {}".format(results)
    elif args.subparser_name == "replay":
        raise NotImplementedError
        replay_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()

