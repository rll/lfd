#!/usr/bin/env python

from __future__ import division

import pprint
import argparse
from core import demonstration, registration, transfer, sim_util
from core.constants import ROPE_RADIUS, MAX_ACTIONS_TO_TRY

from core.demonstration import SceneState, GroundTruthRopeSceneState, AugmentedTrajectory, Demonstration
from core.simulation import DynamicSimulationRobotWorld
from core.simulation_object import XmlSimulationObject, BoxSimulationObject, CylinderSimulationObject, RopeSimulationObject
from core.environment import LfdEnvironment, GroundTruthRopeLfdEnvironment, GroundTruthBoxLfdEnvironment
from core.registration import TpsnRpmGTRegistrationFactory, TpsRpmBijRegistrationFactory, TpsRpmRegistrationFactory, TpsSegmentRegistrationFactory, GpuTpsRpmBijRegistrationFactory, GpuTpsRpmRegistrationFactory, TpsnRpmRegistrationFactory, TpsnRegistrationFactory
from core.transfer import PoseTrajectoryTransferer, FingerTrajectoryTransferer
from core.registration_transfer import TwoStepRegistrationAndTrajectoryTransferer, UnifiedRegistrationAndTrajectoryTransferer
from core.action_selection import GreedyActionSelection

from rapprentice import eval_util, util
from rapprentice import tps_registration, planning
 
from rapprentice import berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     tps, func_utils, resampling, ropesim, rope_initialization
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time

import trajoptpy, openravepy
from rapprentice.knot_classifier import isKnot as is_knot, calculateCrossings
import os, os.path, numpy as np, h5py
from rapprentice.util import redprint, yellowprint
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random
import hashlib

class GlobalVars:
    exec_log = None
    actions = None
    actions_cache = None
    demos = None

def get_move_traj(t_start, t_end, start_fixed, lfd_env, lr, rotaxis):
    n_steps = 10
    if rotaxis != [0,0,0,0]:
        T = openravepy.matrixFromAxisAngle(np.array([0,0,np.pi/4]))[:3,:3]
        R = T.dot(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    else:
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    ee_link_name = "%s_gripper_tool_frame"%lr
    ee_link = lfd_env.world.robot.GetLink(ee_link_name)

    hmat_start = np.r_[np.c_[R, t_start], np.c_[0,0,0,1]]
    hmat_end = np.r_[np.c_[R, t_end], np.c_[0,0,0,1]]
    new_hmats = np.asarray(resampling.interp_hmats(np.arange(n_steps), np.r_[0, n_steps-1], [hmat_start, hmat_end]))
    dof_vals = lfd_env.world.robot.GetManipulator(manip_name).GetArmDOFValues()
    old_traj = np.tile(dof_vals, (n_steps,1))
    
    traj, _, _ = planning.plan_follow_traj(lfd_env.world.robot, manip_name, ee_link, new_hmats, old_traj, start_fixed=start_fixed, beta_rot=10000.0)
    return traj, new_hmats

def generate_box_demonstration(lfd_env,box0_pos, box1_pos, move_height, box_depth, sim, rotaxis):
    lr = 'r'
    manip_names = {"l":"leftarm", "r":"rightarm"}

    dof_inds = sim_util.dof_inds_from_name(lfd_env.world.robot, manip_names['r']) + (sim_util.dof_inds_from_name(lfd_env.world.robot, manip_names['l']))

    t1, h1 = get_move_traj(box0_pos + np.r_[0,0,move_height], box0_pos + np.r_[0,0,box_depth/2-0.03], False, lfd_env, 'r', rotaxis)
    t2, h2 = get_move_traj(box0_pos + np.r_[0,0,box_depth/2-0.03], box0_pos + np.r_[0,0,move_height], False, lfd_env, 'r', rotaxis)
    t3, h3 = get_move_traj(box0_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,move_height], False, lfd_env, 'r', rotaxis)
    t4, h4 = get_move_traj(box1_pos + np.r_[0,0,move_height], box1_pos + np.r_[0,0,box_depth/2+box_depth/2-0.02+0.001], False, lfd_env, 'r', rotaxis)
    t5, h5 = get_move_traj(box1_pos + np.r_[0,0,box_depth+box_depth/2-0.02+0.001], box1_pos + np.r_[0,0,move_height], False, lfd_env, 'r', rotaxis)

    full_traj_r = np.r_[t1,t2,t3,t4,t5]

    dof_vals = lfd_env.world.robot.GetManipulator(manip_names['l']).GetArmDOFValues()

    full_traj = np.c_[full_traj_r,np.tile(dof_vals,(50,1))]

    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    R = np.eye(3)

    demonstration_traj = AugmentedTrajectory.create_from_full_traj(lfd_env.world.robot, (full_traj,dof_inds))
    demonstration_traj.lr2open_finger_traj = {'l': np.tile(np.array([False],dtype=bool),50),'r': np.tile(np.array([False],dtype=bool),50)}
    demonstration_traj.lr2close_finger_traj = {'l': np.tile(np.array([False],dtype=bool),50),'r': np.tile(np.array([False],dtype=bool),50)}
    demonstration_traj.lr2finger_traj = {'l': np.tile(np.array([0.4]), (50,1)), 'r': np.tile(np.array([0.4]), (50,1))}
    demonstration_traj.lr2ee_traj = {'r': np.r_[h1,h2,h3,h4,h5],'l': np.tile(np.r_[np.c_[R, np.array([.101,-.682,1.16])],np.c_[0,0,0,1]], (50,1,1))}
    demonstration_traj.lr2finger_traj['r'][10:,] = .4
    demonstration_traj.lr2finger_traj['r'][39:,] = .4
    demonstration_traj.lr2close_finger_traj['r'][9] = True
    demonstration_traj.lr2open_finger_traj['r'][38] = True

    return demonstration_traj

def box_eval_on_holdout(args, reg_and_traj_transferer, lfd_env, sim):
    table_height = 0.78
    box_length = 0.04
    box_depth = 0.12
    x_start_dist = 0.4
    box0_pos = np.r_[x_start_dist, -.25, table_height+box_depth/2]
    box1_pos = np.r_[x_start_dist, 0, table_height+box_depth/2]
    box1_pos_2 = np.r_[.6, 0, table_height+box_depth/2]
    move_height = .3

    x_offset = .15
    #box0 = BoxSimulationObject("box0", box0_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True)
    #sim_objs.append(box0)
    static_offset = 0.007
    #z_offset = box_depth-0.08
    z_offset = 0
    gt=True

    success,failure=0,0
    succeeds = []
    offset=0
    #rotaxis=[0,0,1,45]
    rotaxis=[0,0,0,0]

    if rotaxis!=[0,0,0,0]:
        box2 = BoxSimulationObject("box2", np.r_[x_start_dist + offset + (box_length+static_offset)/np.sqrt(2),0 + (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length/2, box_length*3/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box3 = BoxSimulationObject("box3", np.r_[x_start_dist + offset - (box_length+static_offset)/np.sqrt(2),0 + (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length*3/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box4 = BoxSimulationObject("box4", np.r_[x_start_dist + offset + (box_length+static_offset)/np.sqrt(2),0 - (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length*3/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box5 = BoxSimulationObject("box5", np.r_[x_start_dist + offset - (box_length+static_offset)/np.sqrt(2),0 - (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length/2, box_length*3/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box0 = BoxSimulationObject("box0", box0_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True)
    else:
        box2 = BoxSimulationObject("box2", np.r_[x_start_dist + offset + (box_length+static_offset),0,table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box3 = BoxSimulationObject("box3", np.r_[x_start_dist + offset - (box_length+static_offset),0,table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box4 = BoxSimulationObject("box4", np.r_[x_start_dist + offset, 0 - (box_length+static_offset),table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box5 = BoxSimulationObject("box5", np.r_[x_start_dist + offset, 0 + (box_length+static_offset),table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box0 = BoxSimulationObject("box0", box0_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True)

    sim.add_objects([box0,box2,box3,box4,box5])
    gt=True
    reg_factory=None
    if args.eval.reg_type == 'tpsn':
        reg_factory = TpsnRegistrationFactory(GlobalVars.demos, sim=sim)
    elif args.eval.reg_type == 'tpsnrpm':
        reg_factory = TpsnRpmRegistrationFactory(GlobalVars.demos, sim=sim)
    elif args.eval.reg_type == 'bij':
        reg_factory = TpsRpmBijRegistrationFactory(GlobalVars.demos, n_iter=20)
    elif args.eval.reg_type == 'rpm':
        reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
    elif args.eval.reg_type == 'tpsnrpmgt':
        gt=True
        reg_factory = TpsnRpmGTRegistrationFactory(GlobalVars.demos,sim=sim)

    bt_box0 = sim.bt_env.GetObjectByName("box0")
    T = openravepy.matrixFromAxisAngle(np.array([0,0,np.pi/4]))
    T[:3,3] = bt_box0.GetTransform()[:3,3]
    bt_box0.SetTransform(T)
    d_traj = generate_box_demonstration(lfd_env, box0_pos, box1_pos, move_height, box_depth, sim, rotaxis)
    if gt:
        sc_dem = lfd_env.observe_scene("demonstration",ground_truth=gt)
    else:
        sc_dem, i0 = lfd_env.observe_scene("demonstration",ground_truth=gt)
        reg_factory.i0=i0
    d1 = Demonstration("d1",sc_dem,d_traj)
    """
    bt_box2 = sim.bt_env.GetObjectByName("box2")
    bt_box3 = sim.bt_env.GetObjectByName("box3")
    bt_box4 = sim.bt_env.GetObjectByName("box4")
    bt_box5 = sim.bt_env.GetObjectByName("box5")
    T = openravepy.matrixFromAxisAngle(np.array([0,0,np.pi/4]))
    T[:3,3] = bt_box2.GetTransform()[:3,3]
    bt_box2.SetTransform(T)
    sim.update()
    bt_box3.SetTransform(T)
    sim.update()
    bt_box4.SetTransform(T)
    sim.update()
    bt_box5.SetTransform(T)
    sim.update()
    """
    lfd_env.box0pos = box0_pos
    lfd_env.box1pos = box1_pos

    #traj_transferer = PoseTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost)
    
    traj_transferer = FingerTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.gamma, args.eval.use_collision_cost)
    a,count=0,0
    #for b in np.linspace(1e-7,1e-13,15):
        #for n in np.linspace(1e6,1e8,15):
    #for b,n in [(4.28572e-08, 43428571.42857143)]:
    n=0
    box_offset = 0
    #for b in np.linspace(1e5,1e-20,20):
    #for b in np.linspace(1e-7,1e-13,5):
        #for n in np.linspace(1e6,1e8,5):
    #for b_init in np.linspace(1e10,1e-2,10):
    """
    for n_init in np.linspace(1e-8,1e-8,1):
        for n_final in np.linspace(3e-4,3e-4,1):
            for b_init in np.linspace(1e2,1e2,1):
                for b_final in np.linspace(1e-3,1e-3,1):
    """
    
    #for b_final in np.linspace(1e-1,1e-5,10):
        #for n_final in np.linspace(1e-2,1e-7,10):
    z = open("tpsrpm_gt_9_29_2014.txt",'w')
    #for b in np.linspace(1,1e-30,100):
    reg_factory.bend_coef=1e-20
    for offset in np.linspace(.2,.0,1000):
        sim_util.reset_arms_to_side(lfd_env.sim)
        sim.remove_objects([box2,box3,box4,box5,box0])

        """
        box2 = BoxSimulationObject("box2", np.r_[x_start_dist + box_length1,0,table_height+box_depth/2-static_offset], [box_length/2, box_length, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box3 = BoxSimulationObject("box3", np.r_[x_start_dist - box_length1,0,table_height+box_depth/2-static_offset], [box_length/2, box_length, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box4 = BoxSimulationObject("box4", np.r_[x_start_dist, -box_length1,table_height+box_depth/2-static_offset], [box_length, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        box5 = BoxSimulationObject("box5", np.r_[x_start_dist, box_length1 ,table_height+box_depth/2-static_offset], [box_length, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
        """

        if rotaxis!=[0,0,0,0]:
            box2 = BoxSimulationObject("box2", np.r_[x_start_dist + offset + (box_length+static_offset)/np.sqrt(2),0 + (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length/2, box_length*3/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box3 = BoxSimulationObject("box3", np.r_[x_start_dist + offset - (box_length+static_offset)/np.sqrt(2),0 + (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length*3/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box4 = BoxSimulationObject("box4", np.r_[x_start_dist + offset + (box_length+static_offset)/np.sqrt(2),0 - (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length*3/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box5 = BoxSimulationObject("box5", np.r_[x_start_dist + offset - (box_length+static_offset)/np.sqrt(2),0 - (box_length+static_offset)/np.sqrt(2),table_height+box_depth/2-z_offset], [box_length/2, box_length*3/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box0 = BoxSimulationObject("box0", box0_pos+np.array([box_offset,0,0]), [box_length/2, box_length/2, box_depth/2], dynamic=True)
        else:
            box2 = BoxSimulationObject("box2", np.r_[x_start_dist + offset + (box_length+static_offset),0,table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box3 = BoxSimulationObject("box3", np.r_[x_start_dist + offset - (box_length+static_offset),0,table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box4 = BoxSimulationObject("box4", np.r_[x_start_dist + offset,0 - (box_length+static_offset),table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box5 = BoxSimulationObject("box5", np.r_[x_start_dist + offset,0 + (box_length+static_offset),table_height+box_depth/2-z_offset], [box_length/2, box_length/2, box_depth/2], dynamic=False, rotationaxis=rotaxis)
            box0 = BoxSimulationObject("box0", box0_pos+np.array([box_offset,0,0]), [box_length/2, box_length/2, box_depth/2], dynamic=True)

        lfd_env.box1pos_2=np.array([x_start_dist+offset, 0])

        sim.add_objects([box2,box3,box4,box5,box0])
        sim.update()
        sim.viewer.Step()

        if rotaxis != [0,0,0,0]:
            bt_box0 = sim.bt_env.GetObjectByName("box0")
            T = openravepy.matrixFromAxisAngle(np.array([0,0,np.pi/4]))
            T[:3,3] = bt_box0.GetTransform()[:3,3]
            bt_box0.SetTransform(T)

        sim.update()
        sim.viewer.Step()
        #reg_factory.bend_coef=
        #reg_factory.normal_coef=
        #reg_factory.bend_coef=b
        #reg_factory.bend_coef_init=1e6
        #reg_factory.bend_coef_init=b_init
        #reg_factory.bend_coef_final=b_final
        #reg_factory.normal_coef_init=n_init
        #reg_factory.normal_coef_final=n_final

        #reg_factory.bend_coef_final = 1.0000000000000001e-05
        #reg_factory.normal_coef_final = 9.9999999999999995e-08
        if gt:
            sc_test = lfd_env.observe_scene("test",ground_truth=gt)
        else:
            sc_test,i1 = lfd_env.observe_scene("test",ground_truth=gt)
            reg_factory.i1=i1
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
        test_aug_traj = reg_and_traj_transferer.transfer(d1, sc_test, plotting=args.plotting)
        #lfd_env.execute_augmented_trajectory(d1.aug_traj,step_viewer=args.animation, interactive=args.interactive)
        lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive)

        bt_box0 = lfd_env.sim.bt_env.GetObjectByName('box0')
        final_pos = bt_box0.GetTransform()[:3,3]
        sim.settle()
        if final_pos[1] < (box1_pos_2[1] + box_length/2) and final_pos[1] > (box1_pos_2[1] - box_length/2) and final_pos[0] < (box1_pos[0]+offset + box_length/2) and final_pos[0] > (box1_pos[0]+offset - box_length/2) and final_pos[2] < table_height+box_depth/2+.04:
            success+=1
            #ipy.embed()
            print str(len(succeeds)) + "\n\n-------SUCCESS--------(" + str(success)+"/" + str(success+failure) + ")\n\n"
            #succeeds.append((b,1))
            z.write(str(offset)+","+str(1)+"\n")
            #succeeds.append((b_final,n_final,offset))
        else:
            failure+=1
            #break
            print str(len(succeeds)) + "\n\n-------FAILURE--------(" + str(success)+"/" + str(success+failure) + ")\n\n" 
            #succeeds.append((offset,0))
            z.write(str(offset)+","+str(0)+"\n")
        sim.remove_objects([box0])
        sim.add_objects([box0])
        #sim.remove_objects([box0])
        #sim.add_objects([box0])
        #if success==5:
            #succeeds.append((b_final,n_final))
    #z.close()
    ipy.embed()
    print success,failure
    

def eval_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim):
    """TODO
    
    Args:
        action_selection: ActionSelection
        reg_and_traj_transferer: RegistrationAndTrajectoryTransferer
        lfd_env: LfdEnvironment
        sim: DynamicSimulation
    """
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_indexed_items(holdoutfile, task_list=args.tasks, task_file=args.taskfile, i_start=args.i_start, i_end=args.i_end)

    rope_params = sim_util.RopeParams()
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        redprint("task %s" % i_task)
        init_rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        rope = RopeSimulationObject("rope", init_rope_nodes, rope_params)

        sim.add_objects([rope])
        sim.settle(step_viewer=args.animation)
        
        for i_step in range(args.eval.num_steps):
            redprint("task %s step %i" % (i_task, i_step))
            
            sim_util.reset_arms_to_side(sim)
            if args.animation:
                sim.viewer.Step()
            sim_state = sim.get_state()
            sim.set_state(sim_state)
            scene_state = lfd_env.observe_scene()

            # plot cloud of the test scene
            handles = []
            if args.plotting:
                handles.append(sim.env.plot3(scene_state.cloud[:,:3], 2, scene_state.color if scene_state.color is not None else (0,0,1)))
                sim.viewer.Step()
            
            eval_stats = eval_util.EvalStats()
            
            start_time = time.time()
            try:
                agenda, q_values_root = action_selection.plan_agenda(scene_state)
            except ValueError: #e.g. if cloud is empty - any action is hopeless
                break
            eval_stats.action_elapsed_time += time.time() - start_time
            
            eval_stats.generalized = True
            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1
            for i_choice in range(num_actions_to_try):
                if q_values_root[i_choice] == -np.inf: # none of the demonstrations generalize
                    eval_stats.generalized = False
                    break
                redprint("TRYING %s"%agenda[i_choice])

                best_root_action = agenda[i_choice]

                start_time = time.time()
                test_aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], scene_state, plotting=args.plotting)
                eval_stats.feasible, eval_stats.misgrasp = lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive, check_feasible=args.eval.check_feasible)
                eval_stats.exec_elapsed_time += time.time() - start_time
                
                if not args.eval.check_feasible or eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action and we care about feasibility
                     break
                else:
                     sim.set_state(sim_state)
            print "BEST ACTION:", best_root_action

            knot = is_knot(rope.rope.GetControlPoints())
            results = {'scene_state':scene_state, 'best_action':best_root_action, 'values':q_values_root, 'aug_traj':test_aug_traj, 'eval_stats':eval_stats, 'sim_state':sim_state, 'knot':knot}
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, results)
            
            if not eval_stats.generalized:
                assert not knot
                break
            
            if args.eval.check_feasible and not eval_stats.feasible:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                assert not knot
                break
            
            if knot:
                num_successes += 1
                break;
        
        sim.remove_objects([rope])
        
        num_total += 1
        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def eval_on_holdout_parallel(args, action_selection, transfer, lfd_env, sim):
    raise NotImplementedError
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_indexed_items(holdoutfile, task_list=args.tasks, task_file=args.taskfile, i_start=args.i_start, i_end=args.i_end)

    rope_params = sim_util.RopeParams()
    if args.eval.rope_param_radius is not None:
        rope_params.radius = args.eval.rope_param_radius
    if args.eval.rope_param_angStiffness is not None:
        rope_params.angStiffness = args.eval.rope_param_angStiffness

    batch_transfer_simulate = BatchTransferSimulate(transfer, lfd_env)

    states = {}
    q_values_roots = {}
    best_root_actions = {}
    state_id2i_task = {}
    results = {}
    successes = {}
    for i_step in range(args.eval.num_steps):
        for i_task, demo_id_rope_nodes in holdout_items:
            if i_task in successes:
                # task already finished
                continue

            redprint("task %s step %i" % (i_task, i_step))

            if i_step == 0:
                sim_util.reset_arms_to_side(lfd_env)

                init_rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
                lfd_env.set_rope_state(RopeState(init_rope_nodes, rope_params))
                states[i_task] = {}
                states[i_task][i_step] = lfd_env.observe_scene(**vars(args.eval))
                best_root_actions[i_task] = {}
                q_values_roots[i_task] = {}
                results[i_task] = {}
                
                if args.animation:
                    lfd_env.viewer.Step()
            
            state = states[i_task][i_step]

            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1

            agenda, q_values_root = select_best(args.eval, state, batch_transfer_simulate) # TODO fix select_best to handle batch_transfer_simulate
            q_values_roots[i_task][i_step] = q_values_root

            i_choice = 0
            if q_values_root[i_choice] == -np.inf: # none of the demonstrations generalize
                successes[i_task] = False
                continue

            best_root_action = agenda[i_choice]
            best_root_actions[i_task][i_step] = best_root_action

            next_state_id = SceneState.get_unique_id()
            batch_transfer_simulate.queue_transfer_simulate(state, best_root_action, next_state_id)

            state_id2i_task[next_state_id] = i_task

        batch_transfer_simulate.wait_while_queue_is_nonempty()
        for result in batch_transfer_simulate.get_results():
            i_task = state_id2i_task[result.state.id]
            results[i_task][i_step] = result
        
        for i_task, demo_id_rope_nodes in holdout_items:
            if i_task in successes:
                # task already finished
                continue

            result = results[i_task][i_step]
            eval_stats = eval_util.EvalStats()
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs, next_state = result.success, result.feasible, result.misgrasp, result.full_trajs, result.state
            # TODO eval_stats.exec_elapsed_time

            if not eval_stats.feasible:  # If not feasible, restore state
                next_state = states[i_task][i_step]
            
            state = states[i_task][i_step]
            best_root_action = best_root_actions[i_task][i_step]
            q_values_root = q_values_roots[i_task][i_step]
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, state, best_root_action, q_values_root, full_trajs, next_state, eval_stats, new_cloud_ds=state.cloud, new_rope_nodes=state.rope_nodes)
            
            states[i_task][i_step+1] = next_state
            
            if not eval_stats.feasible:
                successes[i_task] = False
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                continue
            
            if is_knot(next_state.rope_nodes):
                successes[i_task] = True
                continue
        
        if i_step == args.eval.num_steps - 1:
            for i_task, demo_id_rope_nodes in holdout_items:
                if i_task not in successes:
                    # task ran out of steps
                    successes[i_task] = False

        num_successes = np.sum(successes.values())
        num_total = len(successes)
        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def replay_on_holdout(args, action_selection, transfer, lfd_env, sim):
    loadresultfile = h5py.File(args.replay.loadresultfile, 'r')
    loadresult_items = eval_util.get_indexed_items(loadresultfile, task_list=args.tasks, task_file=args.taskfile, i_start=args.i_start, i_end=args.i_end)
    
    num_successes = 0
    num_total = 0
    
    for i_task, task_info in loadresult_items:
        redprint("task %s" % i_task)

        for i_step in range(len(task_info)):
            redprint("task %s step %i" % (i_task, i_step))
            
            replay_results = eval_util.load_task_results_step(args.replay.loadresultfile, i_task, i_step)
            sim_state = replay_results['sim_state']

            if i_step > 0: # sanity check for reproducibility
                sim_util.reset_arms_to_side(sim)
                if sim.simulation_state_equal(sim_state, sim.get_state()):
                    yellowprint("Reproducible results OK")
                else:
                    yellowprint("The replayed simulation state doesn't match the one from the result file")
                
            sim.set_state(sim_state)

            if args.replay.simulate_traj_steps is not None and i_step not in args.replay.simulate_traj_steps:
                continue
            
            if i_step in args.replay.compute_traj_steps: # compute the trajectory in this step
                best_root_action = replay_results['best_action']
                scene_state = replay_results['scene_state']
                # plot cloud of the test scene
                handles = []
                if args.plotting:
                    handles.append(sim.env.plot3(scene_state.cloud[:,:3], 2, scene_state.color if scene_state.color is not None else (0,0,1)))
                    sim.viewer.Step()
                test_aug_traj = reg_and_traj_transferer.transfer(GlobalVars.demos[best_root_action], scene_state, plotting=args.plotting)
            else:
                test_aug_traj = replay_results['aug_traj']
            feasible, misgrasp = lfd_env.execute_augmented_trajectory(test_aug_traj, step_viewer=args.animation, interactive=args.interactive, check_feasible=args.eval.check_feasible)
            
            if replay_results['knot']:
                num_successes += 1
        
        num_total += 1
        redprint('REPLAY Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="animates if it is non-zero. the viewer is stepped according to this number")
    parser.add_argument("--plotting", type=int, default=1, help="plots if animation != 0 and plotting != 0")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")
    parser.add_argument("--box", type=int, default=0)

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='../.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='../.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")

    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str, nargs='?', default='../bigdata/misc/overhand_actions.h5')
    parser_eval.add_argument('holdoutfile', type=str, nargs='?', default='../bigdata/misc/holdout_set_Jun20_0.10.h5')

    parser_eval.add_argument("transferopt", type=str, nargs='?', choices=['pose', 'finger'], default='finger')
    parser_eval.add_argument("reg_type", type=str, choices=['segment', 'tpsnrpmgt', 'rpm', 'bij','tpsn','tpsnrpm'], default='bij')
    parser_eval.add_argument("--unified", type=int, default=0)
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--downsample_size", type=float, default=0.025)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    parser_eval.add_argument("--ground_truth", type=int, default=1)
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    
    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--check_feasible", type=int, default=0)

    parser_eval.add_argument("--alpha", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=100.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--use_collision_cost", type=int, default=1)

    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_param_radius", type=str, default=None)
    parser_eval.add_argument("--rope_param_angStiffness", type=str, default=None)
    
    parser_eval.add_argument("--parallel", action="store_true")
    parser_eval.add_argument("--gpu", action="store_true", default=False)

    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--compute_traj_steps", type=int, default=[], nargs='*', metavar='i_step', help="recompute trajectories for the i_step of all tasks")
    parser_replay.add_argument("--simulate_traj_steps", type=int, default=None, nargs='*', metavar='i_step', 
                               help="if specified, restore the rope state from file and then simulate for the i_step of all tasks")
                               # if not specified, the rope state is not restored from file, but it is as given by the sequential simulation

    args = parser.parse_args()
    if not args.animation:
        args.plotting = 0
    return args

def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args):
    if args.random_seed is not None: np.random.seed(args.random_seed)
    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')
    
    GlobalVars.demos = {}
    for action, seg_info in GlobalVars.actions.iteritems():
        if args.eval.ground_truth:
            rope_nodes = seg_info['rope_nodes'][()]
            scene_state = GroundTruthRopeSceneState(rope_nodes, ROPE_RADIUS, upsample=args.eval.upsample, upsample_rad=args.eval.upsample_rad, downsample_size=args.eval.downsample_size)
        else:
            full_cloud = seg_info['cloud_xyz'][()]
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
#             # opening_inds/closing_inds are indices before the opening/closing happens, so increment those indices (if they are not out of bound)
#             opening_inds = np.clip(opening_inds+1, 0, len(lr2finger_traj[lr])-1) # TODO figure out if +1 is necessary
#             closing_inds = np.clip(closing_inds+1, 0, len(lr2finger_traj[lr])-1)
            lr2open_finger_traj[lr][opening_inds] = True
            lr2close_finger_traj[lr][closing_inds] = True
        aug_traj = AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
        demo = Demonstration(action, scene_state, aug_traj)
        GlobalVars.demos[action] = demo

def setup_lfd_environment_sim(args):
    actions = h5py.File(args.eval.actionfile, 'r')
    
    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, args.eval.fake_data_segment, args.eval.fake_data_transform) 
    table_height = init_rope_xyz[:,2].mean() - .02
    
    sim_objs = []
    sim_objs.append(XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False))
    sim_objs.append(BoxSimulationObject("table", [1, 0, table_height + (-.1 + .01)], [.85, .85, .1], dynamic=False))

    #BOX PARAMETERS
    table_height = 0.78
    box_length = 0.04
    box_depth = 0.12
    x_start_dist = 0.4
    box0_pos = np.r_[x_start_dist, -.25, table_height+box_depth/2]
    box1_pos = np.r_[x_start_dist, 0, table_height+box_depth/2]
    box1_pos_2 = np.r_[.6, 0, table_height+box_depth/2]
    move_height = .3

    x_offset = .15
    #box0 = BoxSimulationObject("box0", box0_pos, [box_length/2, box_length/2, box_depth/2], dynamic=True)
    #sim_objs.append(box0)
    static_offset = 0.008
    #z_offset = box_depth-0.08
    z_offset = 0
        
    if 'bookshelve' in args.eval.obstacles:
        sim_objs.append(XmlSimulationObject("../data/bookshelve.env.xml", dynamic=False))
    if 'boxes' in args.eval.obstacles:
        sim_objs.append(BoxSimulationObject("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12], dynamic=False))
        sim_objs.append(BoxSimulationObject("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08], dynamic=False))
    if 'cylinders' in args.eval.obstacles:
        sim_objs.append(CylinderSimulationObject("cylinder0", [.7,.43,table_height+(.01+.5)], .12, 1., dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder1", [.7,-.43,table_height+(.01+.5)], .12, 1., dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder2", [.4,.2,table_height+(.01+.65)], .06, .5, dynamic=False))
        sim_objs.append(CylinderSimulationObject("cylinder3", [.4,-.2,table_height+(.01+.65)], .06, .5, dynamic=False))
    
    sim = DynamicSimulationRobotWorld()
    world = sim
    sim.add_objects(sim_objs)
    if args.eval.ground_truth:
        if args.box:
            lfd_env = GroundTruthBoxLfdEnvironment(sim, world, box0_pos[:2], box1_pos[:2], box1_pos[:2] + np.array([x_offset, 0]),box_length/2, table_height+box_depth, table_height, downsample_size=args.eval.downsample_size)
        else:
            lfd_env = GroundTruthRopeLfdEnvironment(sim, world, upsample=args.eval.upsample, upsample_rad=args.eval.upsample_rad, downsample_size=args.eval.downsample_size)
    else:
        lfd_env = LfdEnvironment(sim, world, downsample_size=args.eval.downsample_size)

    dof_inds = sim_util.dof_inds_from_name(sim.robot, '+'.join(init_joint_names))
    values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(init_joint_values, dof_inds) if dof_ind != -1])
    sim.robot.SetDOFValues(values, dof_inds) # this also sets the torso (torso_lift_joint) to the height in the data
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
    
    if args.eval.dof_limits_factor != 1.0:
        assert 0 < args.eval.dof_limits_factor and args.eval.dof_limits_factor <= 1.0
        active_dof_indices = sim.robot.GetActiveDOFIndices()
        active_dof_limits = sim.robot.GetActiveDOFLimits()
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            dof_inds = sim.robot.GetManipulator(manip_name).GetArmIndices()
            limits = np.asarray(sim.robot.GetDOFLimits(dof_inds))
            limits_mean = limits.mean(axis=0)
            limits_width = np.diff(limits, axis=0)
            new_limits = limits_mean + args.eval.dof_limits_factor * np.r_[-limits_width/2.0, limits_width/2.0]
            for i, ind in enumerate(dof_inds):
                active_dof_limits[0][active_dof_indices.tolist().index(ind)] = new_limits[0,i]
                active_dof_limits[1][active_dof_indices.tolist().index(ind)] = new_limits[1,i]
        sim.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])
    return lfd_env, sim

def setup_registration_and_trajectory_transferer(args, sim):
    if args.eval.gpu:
        if args.eval.reg_type == 'rpm':
            reg_factory = GpuTpsRpmRegistrationFactory(GlobalVars.demos, args.eval.actionfile)
        elif args.eval.reg_type == 'bij':
            reg_factory = GpuTpsRpmBijRegistrationFactory(GlobalVars.demos, args.eval.actionfile)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)
    else:
        if args.eval.reg_type == 'segment':
            reg_factory = TpsSegmentRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'rpm':
            reg_factory = TpsRpmRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'bij':
            reg_factory = TpsRpmBijRegistrationFactory(GlobalVars.demos, n_iter=20) #TODO
        elif args.eval.reg_type == 'tpsn':
            reg_factory = TpsnRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'tpsnrpm':
            reg_factory = TpsnRpmRegistrationFactory(GlobalVars.demos)
        elif args.eval.reg_type == 'tpsnrpmgt':
            reg_factory = TpsnRpmGTRegistrationFactory(GlobalVars.demos)
        else:
            raise RuntimeError("Invalid reg_type option %s"%args.eval.reg_type)

    if args.eval.transferopt == 'pose' or args.eval.transferopt == 'finger':
        traj_transferer = PoseTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.beta_rot, args.eval.gamma, args.eval.use_collision_cost)
        if args.eval.transferopt == 'finger':
            traj_transferer = FingerTrajectoryTransferer(sim, args.eval.beta_pos, args.eval.gamma, args.eval.use_collision_cost, init_trajectory_transferer=traj_transferer)
    else:
        raise RuntimeError("Invalid transferopt option %s"%args.eval.transferopt)
    
    if args.eval.unified:
        reg_and_traj_transferer = UnifiedRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    else:
        reg_and_traj_transferer = TwoStepRegistrationAndTrajectoryTransferer(reg_factory, traj_transferer)
    return reg_and_traj_transferer

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
    
    set_global_vars(args)
    trajoptpy.SetInteractive(args.interactive)
    lfd_env, sim = setup_lfd_environment_sim(args)
    reg_and_traj_transferer = setup_registration_and_trajectory_transferer(args, sim)
    action_selection = GreedyActionSelection(reg_and_traj_transferer.registration_factory)

    if args.subparser_name == "eval":
        start = time.time()
        if args.eval.parallel:
            eval_on_holdout_parallel(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
        else:
            if args.box:
                box_eval_on_holdout(args, reg_and_traj_transferer, lfd_env, sim)
            else:
                eval_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
        print "eval time is:\t{}".format(time.time() - start)
    elif args.subparser_name == "replay":
        replay_on_holdout(args, action_selection, reg_and_traj_transferer, lfd_env, sim)
    else:
        raise RuntimeError("Invalid subparser name")

if __name__ == "__main__":
    main()