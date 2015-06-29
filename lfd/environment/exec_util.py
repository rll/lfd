# Contains useful functions for PR2 execution
# The purpose of this class is to eventually consolidate
# the various instantiations of do_task.py

import openravepy, numpy as np, rospy
from rapprentice.yes_or_no import yes_or_no
from rapprentice import animate_traj, math_utils as mu, plotting_openrave, pr2_trajectories, resampling
from lfd.rapprentice import planning
from lfd.environment import sim_util
from lfd.visfeatures import misc_util
from lfd.rapprentice.util import redprint

import roslib
roslib.load_manifest('joint_states_listener_arms')
from joint_states_listener_arms.srv import ReturnJointStates

#JOINT_LENGTH_PER_STEP = 0.05
JOINT_LENGTH_PER_STEP = 0.05
L_HAS_OBJ_MIN = 0.0010  # adjust for thinner/thicker towel
R_HAS_OBJ_MIN = 0.0005
HAS_OBJ_MAX = 0.03

def has_object(lr_arm):
    resp = False
    try:
        get_joint = rospy.ServiceProxy("return_joint_states",ReturnJointStates)
        resp = get_joint(["%s_gripper_joint"%lr_arm])
    except rospy.ServiceException,e:
        rospy.loginfo("Service Call Failed: %s"%e)
        rospy.sleep(0.2)
        return has_object(lr_arm)
    if not resp:
        return False

    print "Position of ", lr_arm, " gripper: ", resp.position
    print "Position of ", lr_arm, " gripper: ", get_joint(["%s_gripper_joint"%lr_arm]).position

    if lr_arm == 'l':
        return L_HAS_OBJ_MIN < resp.position[0] < HAS_OBJ_MAX
    else:
        return R_HAS_OBJ_MIN < resp.position[0] < HAS_OBJ_MAX

def binarize_gripper(angle):
    open_angle = .08
    closed_angle = 0    
    thresh = .04
    if angle > thresh: return open_angle
    else: return closed_angle

def set_gripper_maybesim(lr, value, pr2, robot, execute=1):
    if execute:
        gripper = {"l":pr2.lgrip, "r":pr2.rgrip}[lr]
        gripper.set_angle(value)
        pr2.join_all()

        #import time
        #time.sleep(0.2)
        #if value == 0 and not has_object(lr):
        #    return False
    else:
        robot.SetDOFValues([value*5], [robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()])
    return True

def exec_traj_maybesim(bodypart2traj, pr2, robot, args, execute=1):
    if args.animation:
        dof_inds = []
        trajs = []
        for (part_name, traj) in bodypart2traj.items():
            manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
            dof_inds.extend(robot.GetManipulator(manip_name).GetArmIndices())            
            trajs.append(traj)
        full_traj = np.concatenate(trajs, axis=1)
        robot.SetActiveDOFs(dof_inds)

        animate_traj.animate_traj(full_traj, robot, restore=False, pause=True)
        #animate_traj.animate_traj(full_traj, robot, restore=False, pause=False)

    if execute:
        if not args.prompt or yes_or_no("execute?"):
            pr2_trajectories.follow_body_traj(pr2, bodypart2traj)
        else:
            return False

    return True

def get_ee_traj_from_joint_traj(sim_env, lr, joint_or_full_traj):
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    ee_link_name = "%s_gripper_tool_frame"%lr
    ee_link = sim_env.robot.GetLink(ee_link_name)
    if type(joint_or_full_traj) == tuple: # it is a full_traj
        joint_traj = joint_or_full_traj[0]
        dof_inds = joint_or_full_traj[1]
    else:
        joint_traj = joint_or_full_traj
        dof_inds = sim_env.robot.GetManipulator(manip_name).GetArmIndices()
    ee_traj = []
    with openravepy.RobotStateSaver(sim_env.robot):
        for i_step in range(joint_traj.shape[0]):
            sim_env.robot.SetDOFValues(joint_traj[i_step], dof_inds)
            ee_traj.append(ee_link.GetTransform())
    return np.array(ee_traj)

def execute_traj(sim_env, seg_name, seg_info, handles, f, old_xyz, new_xyz, args, execute=1, gamma = 1000.0):
    link2eetraj = {}
    for lr in 'lr':
        link_name = "%s_gripper_tool_frame"%lr
        old_ee_traj = np.asarray(seg_info[link_name]["hmat"])
        new_ee_traj = f.transform_hmats(old_ee_traj)
        link2eetraj[link_name] = new_ee_traj
            
        handles.append(sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
        handles.append(sim_env.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))
            
    handles.extend(plotting_openrave.draw_grid(sim_env.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04)) 
    
    miniseg_starts, miniseg_ends = sim_util.split_trajectory_by_gripper(seg_info)    
    success = True
    redprint("mini segments:")
    print '\t', miniseg_starts, miniseg_ends
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):
            
        #if args.execution=="real": sim_env.pr2.update_rave()
        if execute: sim_env.pr2.update_rave()

        ################################    
        redprint("Generating joint trajectory for segment %s, part %i"%(seg_name, i_miniseg))
        print "Gamma in execute_traj", gamma
            
        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = np.asarray(seg_info[manip_name][i_start:i_end+1])
            if sim_util.arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            print "old traj shape:", old_total_traj.shape[0]
            if old_total_traj.shape[0] < 30:
                print "resampling more"
                _, timesteps_rs = sim_util.unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP / 2.0)
            else:
                _, timesteps_rs = sim_util.unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
            print "Timesteps:", len(timesteps_rs)
       
        ### Generate fullbody traj
        bodypart2traj = {}            
        print "\t lr2oldtraj: ", lr2oldtraj.items()
        for (lr,old_joint_traj) in lr2oldtraj.items():

            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                 
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)

            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = link2eetraj[ee_link_name][i_start:i_end+1]          
            print "\t Resampling trajectory"

            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            if execute: sim_env.pr2.update_rave()

            print "\t Planning trajectory"
            oldstdout_fno, oldstderr_fno = misc_util.suppress_stdout()
            new_joint_traj, obj_value, pose_costs = planning.plan_follow_traj(sim_env.robot, manip_name,
                             sim_env.robot.GetLink(ee_link_name), new_ee_traj_rs, old_joint_traj_rs, gamma = gamma * 1.0)
            misc_util.unsuppress_stdout(oldstdout_fno, oldstderr_fno)

            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj

            print "\t Getting list of hmats"
            # Get list of hmats for gripper trajectory after TrajOpt --> Visualize as line
            trajopt_ee_traj = get_ee_traj_from_joint_traj(sim_env, lr, new_joint_traj)
            handles.append(sim_env.env.drawlinestrip(trajopt_ee_traj[:,:3,3], 2, (0,0,1,1)))
                
        ################################    
        redprint("Executing joint trajectory for segment %s, part %i using arms '%s'"%(seg_name, i_miniseg, bodypart2traj.keys()))

        for lr in 'lr':
            success &= set_gripper_maybesim(lr, binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start]), sim_env.pr2, sim_env.robot)
            # Doesn't actually check if grab occurred, unfortunately

        if not success:
            print "UNSUCCESSFUL GRASP"
            break
        
        if len(bodypart2traj) > 0:
            success &= exec_traj_maybesim(bodypart2traj, sim_env.pr2, sim_env.robot, args)

        if not success: break

    redprint("Segment %s result: %s"%(seg_name, success))
    return success

