from __future__ import division

from core import demonstration, registration, transfer, sim_util
from core.constants import ROPE_RADIUS, MAX_ACTIONS_TO_TRY, DEFAULT_LAMBDA, N_ITER_CHEAP, TORSO_HEIGHT, TORSO_IND
from core.demonstration import GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration, BootstrapDemonstration
from core.simulation import DynamicSimulationRobotWorld
from core.environment import GroundTruthRopeLfdEnvironment
from core.registration import GpuTpsRpmBijRegistrationFactory, loglinspace
from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer
from core.action_selection import SoftmaxActionSelection
from core.file_utils import group_or_dataset_to_obj, add_obj_to_group
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject

from rapprentice.knot_classifier import isKnot as is_knot, calculateCrossings

from tpsopt.cuda_funcs import reset_cuda
import trajoptpy
import os, os.path
from constants import BEND_COEF_DIGITS
from rapprentice import eval_util, util
import numpy as np
import argparse, sys, h5py, pprint

# TRAJOPT CONSTANTS
TRAJOPT_DEFAULTS = {'beta_pos'           : 1000000.0,
                    'beta_rot'           : 100.0,
                    'gamma'              : 1000.0,
                    'use_collision_cost' : True}

def parse_input_args():
    usage = """
    Run {0} --help for a list of arguments
    """.format(sys.argv[0])

    parser = argparse.ArgumentParser(usage=usage)

    parser.add_argument("transfer_type", type=str, choices=['derived-traj', 'derived-correspondence', 'parent-selection'])

    parser.add_argument("resultfile", type=str, nargs= '?', default='bigdata/bootstrap/bootstrap_res.h5')
    parser.add_argument("trainfile", type=str, nargs = '?', default='bigdata/bootstrap/sept_13_0.1_train_0.h5')
    parser.add_argument("actionfile", type=str, nargs='?', default='bigdata/misc/overhand_actions.processed_ground_truth.h5')



    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='../.win_prop.txt')


    parser.add_argument('--bend_coef_init', type=float, default=DEFAULT_LAMBDA[0])
    parser.add_argument('--bend_coef_final', type=float, default=DEFAULT_LAMBDA[1])
    parser.add_argument('--n_iter', type=int, default=N_ITER_CHEAP)

    parser.add_argument("--num_steps", type=int, default=5)

    parser.add_argument("--train_sizes", type=int, nargs="+", default=None,
                        help="A space separated list of the number of bootstrapping iterations each bootstrap file should be created from")
    parser.add_argument("--plotting", type=int, default=1, help="plots if animation != 0 and plotting != 0")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()
    if not args.animation: args.plotting = 0
    return args

class GlobalVars:
    demos = None
    lfd_env = None
    sim = None
    bend_coefs = None
    ID = 0
    n_orig_transfers = 0
    n_transfers = 0

        
def initialize(args):           
    print "reading actionfile"
    try:
        actf = h5py.File(args.actionfile)
        GlobalVars.demos = group_or_dataset_to_obj(actf)
    finally:
        actf.close()
    print "intializing simulation"
    GlobalVars.bend_coefs = np.around(loglinspace(args.bend_coef_init, args.bend_coef_final, args.n_iter), 
                                      BEND_COEF_DIGITS)
    init_rope_xyz = GlobalVars.demos.values()[0].scene_state.transfer_cld()
    table_height = init_rope_xyz[:,2].mean() - .02
    
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height + (-.1 + .01)], [.85, .85, .1], dynamic=False))

    sim = DynamicSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    #Bootsrapping only uses ground truth
    lfd_env = GroundTruthRopeLfdEnvironment(sim, world)

    # dof_inds = sim_util.dof_inds_from_name(sim.robot, '+'.join(init_joint_names))
    # values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    # sim.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
    sim.robot.SetDOFValues([TORSO_HEIGHT], [TORSO_IND])
    sim_util.reset_arms_to_side(sim)
    
    if args.animation:
        viewer = trajoptpy.GetViewer(sim.env)
        if os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                viewer.SetWindowProp(*window_prop)
                viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = viewer.GetWindowProp()
                camera_matrix = viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        viewer.Step()
    GlobalVars.lfd_env = lfd_env
    GlobalVars.sim = sim
    print "initialization complete"

def add_trace(trace, args):
    n_orig = 0
    for demo in trace:
        try:
            x = demo.parent.parent
        except AttributeError:
            n_orig += 1
        demo.compute_solver_data(GlobalVars.bend_coefs)
        GlobalVars.demos[demo.name] = demo
    print 'added {} orig transfers {} derived'.format(n_orig, len(trace) - n_orig)
    GlobalVars.n_orig_transfers += n_orig
    GlobalVars.n_transfers += len(trace)
    print '{}/{} orig vs derived transfers total'.format(GlobalVars.n_orig_transfers, GlobalVars.n_transfers)
    return compute_action_selector(args)

def compute_action_selector(args):
    # TODO implement mem constrained GPU reg factory
    reset_cuda() # so we have as much free space on the GPU as possible
    if args.transfer_type != 'derived-traj':
        print 'only derived trajcetories are implemented currently'
        raise NotImplementedError('only derived trajcetories are implemented currently')
    reg_factory          = GpuTpsRpmBijRegistrationFactory(GlobalVars.demos)
    init_transferer      = PoseTrajectoryTransferer(GlobalVars.sim, **TRAJOPT_DEFAULTS)
    traj_transferer      = FingerTrajectoryTransferer(GlobalVars.sim, 
                                                      init_trajectory_transferer=init_transferer, 
                                                      **TRAJOPT_DEFAULTS)
    reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    action_selection = SoftmaxActionSelection(reg_factory, alpha = args.alpha)
    
    return action_selection, reg_and_traj_transferer

def do_single_task(rope_nodes, action_selection, reg_and_traj_transferer, args):
    sim = GlobalVars.sim
    lfd_env = GlobalVars.lfd_env
    rope_params = sim_util.RopeParams()
    rope = RopeSimulationObject("rope", rope_nodes, rope_params)
    sim.add_objects([rope])
    sim.settle(step_viewer=args.animation)
    exec_trace = []
    
    for i_step in range(args.num_steps):
        sim_state = sim.get_state()
        sim.set_state(sim_state)
        scene_state = lfd_env.observe_scene()

        try:
            agenda, q_values_root = action_selection.plan_agenda(scene_state)
        except ValueError: #e.g. if cloud is empty - any action is hopeless
            sim.remove_objects([rope])
            return (False,)

        if q_values_root[0] == -np.inf: # none of the demonstrations generalize
            sim.remove_objects([rope])
            return (False,)

        best_root_action = agenda[0]
        test_aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], scene_state, plotting=args.plotting)
        feasible, misgrasp = lfd_env.execute_augmented_trajectory(test_aug_traj, 
                                                                  step_viewer=args.animation, 
                                                                  interactive=args.interactive)
        sim.settle()
        if misgrasp: continue
        exec_trace.append(BootstrapDemonstration(GlobalVars.ID, scene_state, test_aug_traj, 
                                                 parent_demo = GlobalVars.demos[best_root_action]))
        GlobalVars.ID +=1 
        if is_knot(rope.rope.GetControlPoints()):
            sim.remove_objects([rope])
            return (True, exec_trace)
    sim.remove_objects([rope])
    return (False, )

def train(args):
    resultf = h5py.File(args.resultfile, 'w')
    trainf  = h5py.File(args.trainfile, 'r')
    action_selection, reg_and_traj_transferer = compute_action_selector(args)
    for i in range(args.train_sizes[-1]):
        rope_nodes = trainf[str(i)]['rope_nodes'][:]
        exec_result = do_single_task(rope_nodes, action_selection, reg_and_traj_transferer, args)
        if exec_result[0]:
            action_selection, reg_and_traj_transferer = add_trace(exec_result[1], args)            
        if i+1 in args.train_sizes:
            add_obj_to_group(resultf, str(i+1), GlobalVars.demos)
        
def main():
    args = parse_input_args()
    initialize(args)
    train(args)

if __name__ == '__main__':
    main()
