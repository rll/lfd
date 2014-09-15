import rospy
import numpy as np
from rapprentice import conversions as conv, math_utils as mu, \
    kinematics_utils as ku, retiming, PR2

def make_joint_traj(xyzs, quats, manip, ref_frame, targ_frame, filter_options = 0):
    "do ik and then fill in the points where ik failed"

    n = len(xyzs)
    assert len(quats) == n

    robot = manip.GetRobot()
    joint_inds = manip.GetArmIndices()
    robot.SetActiveDOFs(joint_inds)
    orig_joint = robot.GetActiveDOFValues()

    joints = []
    inds = []

    for i in xrange(0,n):
        mat4 = conv.trans_rot_to_hmat(xyzs[i], quats[i])
        joint = PR2.cart_to_joint(manip, mat4, ref_frame, targ_frame, filter_options)
        if joint is not None:
            joints.append(joint)
            inds.append(i)
            robot.SetActiveDOFValues(joint)


    robot.SetActiveDOFValues(orig_joint)


    rospy.loginfo("found ik soln for %i of %i points",len(inds), n)
    if len(inds) > 2:
        joints2 = mu.interp2d(np.arange(n), inds, joints)
        return joints2, inds
    else:
        return np.zeros((n, len(joints))), []



def follow_body_traj(pr2, bodypart2traj, wait=True, base_frame = "/base_footprint", speed_factor=1):

    rospy.loginfo("following trajectory with bodyparts %s", " ".join(bodypart2traj.keys()))

    name2part = {"lgrip":pr2.lgrip,
                 "rgrip":pr2.rgrip,
                 "larm":pr2.larm,
                 "rarm":pr2.rarm,
                 "base":pr2.base,
                 "torso":pr2.torso}
    for partname in bodypart2traj:
        if partname not in name2part:
            raise Exception("invalid part name %s"%partname)


    #### Go to initial positions #######


    # for (name, part) in name2part.items():
    #     if name in bodypart2traj:
    #         part_traj = bodypart2traj[name]
    #         if name == "lgrip" or name == "rgrip":
    #             part.set_angle(np.squeeze(part_traj)[0])
    #         elif name == "larm" or name == "rarm":
    #             part.goto_joint_positions(part_traj[0])
    #         elif name == "base":
    #             part.goto_pose(part_traj[0], base_frame)
    # pr2.join_all()


    #### Construct total trajectory so we can retime it #######


    n_dof = 0
    trajectories = []
    vel_limits = []
    acc_limits = []
    bodypart2inds = {}
    for (name, part) in name2part.items():
        if name in bodypart2traj:
            traj = bodypart2traj[name]
            if traj.ndim == 1: traj = traj.reshape(-1,1)
            trajectories.append(traj)
            vel_limits.extend(part.vel_limits)
            acc_limits.extend(part.acc_limits)
            bodypart2inds[name] = range(n_dof, n_dof+part.n_joints)
            n_dof += part.n_joints

    trajectories = np.concatenate(trajectories, 1)

    vel_limits = np.array(vel_limits)*speed_factor


    times = retiming.retime_with_vel_limits(trajectories, vel_limits)
    times_up = np.linspace(0, times[-1], int(np.ceil(times[-1]/.1)))
    if times_up.size == 0:    # if traj has no motion
        return True
    traj_up = mu.interp2d(times_up, times, trajectories)


    #### Send all part trajectories ###########
    for (name, part) in name2part.items():
        if name in bodypart2traj:
            part_traj = traj_up[:,bodypart2inds[name]]
            if name in {'lgrip', 'rgrip'}:
                part.follow_timed_trajectory(times_up, part_traj.flatten())
            elif name in {'larm', 'rarm', 'torso'}:
                vels = ku.get_velocities(part_traj, times_up, .001)
                part.follow_timed_joint_trajectory(part_traj, vels, times_up)
            elif name == "base":
                part.follow_timed_trajectory(times_up, part_traj, base_frame)

    if wait: pr2.join_all()

    return True


def flatten_compound_dtype(compound_array):
    arrays = []
    for desc in compound_array.dtype.descr:
        field = desc[0]
        arr = compound_array[field]
        if arr.ndim == 1:
            float_arr = arr[:,None].astype('float')
        elif arr.ndim == 2:
            float_arr = arr.astype('float')
        else:
            raise Exception("subarray with field %s must be 1d or 2d"%field)
        arrays.append(float_arr)

    return np.concatenate(arrays, axis=1)

def follow_rave_trajectory(pr2, ravetraj, dof_inds, use_base = False, base_frame="/base_footprint"):

    assert ravetraj.shape[1] == len(dof_inds) + 3*int(use_base)
    bodypart2traj = {}


    rave2ros = {}
    name2part = {"l_gripper":pr2.lgrip,
                 "r_gripper":pr2.rgrip,
                 "l_arm":pr2.larm,
                 "r_arm":pr2.rarm}
    for (partname, part) in name2part.items():
        for (ijoint, jointname) in enumerate(part.joint_names):
            rave2ros[pr2.robot.GetJoint(jointname).GetDOFIndex()] = (partname, ijoint)
        bodypart2traj[partname] = np.repeat(np.asarray(part.get_joint_positions())[None,:], len(ravetraj), axis=0)



    bodypart2used = {}

    for (ravecol, dof_ind) in enumerate(dof_inds):
        if dof_ind in rave2ros:
            partname, partind = rave2ros[dof_ind]
            bodypart2traj[partname][:,partind] = ravetraj[:,ravecol]
        elif dof_ind == pr2.robot.GetJoint("r_gripper_l_finger_joint").GetDOFIndex():
            partname, partind = "r_gripper", 0
            bodypart2traj[partname][:,partind] = ravetraj[:,ravecol]/5.81
        elif dof_ind == pr2.robot.GetJoint("l_gripper_l_finger_joint").GetDOFIndex():
            partname, partind = "l_gripper", 0
            bodypart2traj[partname][:,partind] = ravetraj[:,ravecol]/5.81
        else:
            jointname = pr2.robot.GetJointFromDOFIndex(dof_ind).GetName()
            raise Exception("I don't know how to control this joint %s"%jointname)
        bodypart2used[partname] = True

    for partname in list(bodypart2traj.keys()):
        if partname not in bodypart2used:
            del bodypart2traj[partname]

    if use_base:
        base_traj = ravetraj[:,-3:]
        bodypart2traj["base"] = base_traj

    follow_body_traj2(pr2, bodypart2traj, base_frame = base_frame)
