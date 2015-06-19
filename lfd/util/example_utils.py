#!/usr/bin/env python

import numpy as np
import openravepy

def get_object_limits(obj):
    """
    Returns the bounding box of an openrave body
    Returns: min_x, max_x, min_y, max_y, z
    """
    ab = obj.ComputeAABB()
    max_x = ab.pos()[0] + ab.extents()[0]
    min_x = ab.pos()[0] - ab.extents()[0]

    max_y = ab.pos()[1] + ab.extents()[1]
    min_y = ab.pos()[1] - ab.extents()[1]
    z = ab.pos()[2] + ab.extents()[2]

    return min_x, max_x, min_y, max_y, z

def get_rel_pts(rave_robot):
    """ 
    Buggy
    Get the relative points (four corners) for the 
    rectangular robots
    """
    from itertools import product
    t = openravepy.matrixFromPose([1, 0, 0, 0, 0, 0, 0])
    old_t = rave_robot.GetTransform()
    rave_robot.SetTransform(t)

    min_x, max_x, min_y, max_y, z = get_object_limits(rave_robot)
    all_y_points = np.linspace(min_y, max_y, num = 2, endpoint=True)
    all_x_points = np.linspace(min_x, max_x, num = 2, endpoint=True)
    all_z_points = np.empty((4, 1))
    all_z_points.fill(0)
    rel_pts = np.array(list(product(all_x_points, all_y_points)))
    rel_pts = np.hstack((rel_pts, all_z_points))

    rave_robot.SetTransform(old_t)
    return rel_pts

def get_object_pc(obj, num_x_points, num_y_points):
    """
    Get pointcloud for openrave object
    Very limited: targeted for rectangular objects
    """
    min_x, max_x, min_y, max_y, z = get_object_limits(obj)
    from itertools import product
    y_points = num_y_points 
    x_points = num_x_points 
    total_points = x_points * y_points
    all_y_points = np.linspace(min_y, max_y, num = y_points, endpoint=True)
    all_x_points = np.linspace(min_x, max_x, num = x_points, endpoint=True)
    all_z_points = np.empty((total_points, 1))
    all_z_points.fill(z)
    obj_pc = np.array(list(product(all_x_points, all_y_points)))
    obj_pc = np.hstack((obj_pc, all_z_points))
    return obj_pc

def get_all_obstructions_pc(env):
    """
    Returns the pointcloud for all the obstructions (obstructions should have
    'obstruction' in its name)
    Assumes rectangular obstruction 
    """
    bodies = env.sim.env.GetBodies() 
    obstruction_pc = None
    for body in bodies:
        if "obstruction" in body.GetName():
            curr_obj_pc = get_object_pc(body, 3, 3)
            if obstruction_pc != None:
                obstruction_pc = np.vstack((obstruction_pc, curr_obj_pc))
            else:
                obstruction_pc = curr_obj_pc
    return obstruction_pc

def get_scene_state(env, robot, num_x_points, num_y_points, include_obstruction=False):
    """
    Get current scene state (including robot and depending on the flag, the obstructions)
    """
    robot_kinbody = robot.get_bullet_objects()[0].GetKinBody()
    from itertools import product
    min_x, max_x, min_y, max_y, z = get_object_limits(robot_kinbody)
    y_points = num_y_points 
    x_points = num_x_points 
    total_points = x_points * y_points
    all_y_points = np.linspace(min_y, max_y, num = y_points, endpoint=True)
    all_x_points = np.linspace(min_x, max_x, num = x_points, endpoint=True)
    all_z_points = np.empty((total_points, 1))
    all_z_points.fill(z)
    init_pc = np.array(list(product(all_x_points, all_y_points)))
    init_pc = np.hstack((init_pc, all_z_points))
    
    if include_obstruction:
        obstruction_pc = get_all_obstructions_pc(env)
        init_pc = np.vstack((init_pc, obstruction_pc))
    return init_pc

def generate_pc_from_traj(env, robot, robot_kinbody, traj, obstruction_pc=None,num_x_points = 12, num_y_points = 3, plot=False):

    """
    returns a sequence point clouds, n x k x 3 matrix
    (each pointcloud contains k points)
    """
    # sample points from the robot (initial pc)
    init_pc = get_scene_state(env, robot, num_x_points, num_y_points) 

    init_t = robot_kinbody.GetTransform()
    y_points = num_y_points
    x_points = num_x_points
    total_points = y_points * x_points
    min_x, max_x, min_y, max_y, z = get_object_limits(robot_kinbody)

    # generate pc from trajectory
    if obstruction_pc == None:
        pc_seq = np.empty(((len(traj)), total_points, 3))
        pc_seq[0,:,:] = init_pc
    else:
        pc_seq = np.empty(((len(traj)), total_points + len(obstruction_pc), 3))
        pc_seq[0,:,:] = np.vstack((init_pc, obstruction_pc))
    center_pt = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, z]).reshape(3, 1)
    for i in range(1, len(traj)):
        transform_to_pc = traj[i-1]
        # transform_to_pc[:,3] = transform_to_pc[:,3] - init_t[:,3]
        # transform_to_pc[3,3] = 1
        rotation = transform_to_pc[:3,:3]
        translation = transform_to_pc[:,3] - init_t[:,3]
        translation = translation[:3].reshape(3, 1)
        # incorrect translation
        apply_t = lambda x: np.asarray((np.dot(rotation, x.reshape(3, 1) - center_pt)) + center_pt + translation[:3]).reshape(-1)
        robot_pc = np.array(map(apply_t, init_pc))
        if obstruction_pc == None:
            pc_seq[i,:,:] = robot_pc
        else:
            pc_seq[i,:,:] = np.vstack((robot_pc, obstruction_pc))
    if plot:
        plot_clouds(env, pc_seq)
    return pc_seq

def base_pose_to_mat(traj, z):
    result = np.zeros((len(traj), 4, 4))
    for i in range(len(traj)):
        pose = traj[i]
        x, y, rot = pose
        q = openravepy.quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, z]
        mat = openravepy.matrixFromPose(q + pos)
        result[i,:,:] = mat
    return result

def mat_to_base_pose(traj):
    """
    Untested
    """
    result = np.zeros((len(traj), 3))
    for i in range(len(traj)):
        mat = traj[i]
        pose = openravepy.poseFromMatrix(mat)
        x = pose[4]
        y = pose[5]
        rot = openravepy.axisAngleFromRotationMatrix(mat)[2]
        result[i,:] = np.array([x, y, rot])
    return result

def plot_robot(env, traj_rel_pts, comment="look at robot trajectory plot", show_now=True, colors=[0, 1, 0]):
    handles = []
    if len(traj_rel_pts.shape) == 3:
        traj_rel_pts = traj_rel_pts.reshape((traj_rel_pts.shape[0] * traj_rel_pts.shape[1], 3))
    for i in range(len(traj_rel_pts) // 4):
        index = 4 * i
        lines = traj_rel_pts[index:index+2]
        lines = np.vstack((lines, traj_rel_pts[index + 3]))
        lines = np.vstack((lines, traj_rel_pts[index + 2]))
        lines = np.vstack((lines, traj_rel_pts[index]))
        handles.append(env.sim.env.drawlinestrip(points=lines, linewidth=1.0, colors = np.array(colors)))
        # handles.append(env.sim.env.drawlinestrip(points=np.array(((-0.5, -0.5, 0), (-1.25, 0.5, 0), (-1.5, 1, 0))), linewidth=3.0, colors=np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)))))
    if show_now:
        env.sim.viewer.Step()
        raw_input(comment)
    else:
        return handles

def plot_sampled_demo_pc(env, sampled_pc, comment="look at demo sampled pc", show_now = True, colors=[0, 1, 1]):
    handles = []
    handles.append(env.sim.env.plot3(points = sampled_pc, pointsize=3, colors=colors, drawstyle=1))
    if show_now:
        env.sim.viewer.Step()
        raw_input(comment)
    else:
        return handles

def plot_clouds(env, pc_seqs):
    # for pc in pc_seqs:
    handles = []
    handles.append(env.sim.env.plot3(points = pc_seqs, pointsize=3, colors=[0, 1, 0], drawstyle=1))
    env.sim.viewer.Step()
    raw_input("look at pc")

def draw_two_pc(env, new_pc, f_on_old_pc, include_timesteps=False):
    ##### New: red 
    ##### Old: green
    if include_timesteps:
        new_pc = new_pc[:,:2]
        f_on_old_pc = f_on_old_pc[:,:2]
    new_pc = np.hstack((new_pc, np.zeros((len(new_pc), 1))))
    f_on_old_pc = np.hstack((f_on_old_pc, np.zeros((len(f_on_old_pc), 1))))
    handle1 = plot_sampled_demo_pc(env, new_pc, show_now=False, colors = [1, 0, 0])
    handle2 = plot_sampled_demo_pc(env, f_on_old_pc, show_now=False, colors = [0, 1, 0])
    env.sim.viewer.Step()
    raw_input("Compare two point clouds")

def draw_two_traj(env, new_traj, f_on_old_traj, include_timesteps=False):
    ##### New: red 
    ##### Old: green
    if include_timesteps:
        new_traj = new_traj[:,:2]
        f_on_old_traj = f_on_old_traj[:,:2]
    new_traj = np.hstack((new_traj, np.zeros((len(new_traj), 1))))
    f_on_old_traj = np.hstack((f_on_old_traj, np.zeros((len(f_on_old_traj), 1))))
    handle1 = plot_robot(env, new_traj, show_now=False, colors=[1, 0, 0])
    handle2 = plot_robot(env, f_on_old_traj, show_now=False, colors=[0, 1, 0])
    env.sim.viewer.Step()
    raw_input("Compare two trajectories")


def get_new_pc_from_traj(env, orig_pc_at_origin, time_step_indices, traj_mat, obstruction_pc=None, plot=False):
    """
    Given a trajectory, returns the pointcloud sequence 
    """
    traj = traj_mat[time_step_indices] ## timesteps that matters
    if obstruction_pc == None:
        total_points_per_timestep = orig_pc_at_origin.shape[0]
    else:
        total_points_per_timestep = orig_pc_at_origin.shape[0] + len(obstruction_pc)
    pc_seq = np.zeros(((len(traj)), total_points_per_timestep, 3))

    for i in range(len(traj)):
        transform_to_pc = traj[i]
        rotation = transform_to_pc[:3,:3]
        translation = transform_to_pc[:,3]
        translation = translation[:3].reshape(3, 1)
        apply_t = lambda x: np.asarray(np.dot(rotation, x.reshape(3, 1)) + translation[:3]).reshape(-1)
        if obstruction_pc == None:
            pc_seq[i,:,:] = np.array(map(apply_t, orig_pc_at_origin))
        else:
            pc_seq[i,:,:] = np.vstack((np.array(map(apply_t, orig_pc_at_origin)), obstruction_pc))

        if plot:
            plot_clouds(env, np.array(map(apply_t, orig_pc_at_origin)))

    # returns a two dimension pointcloud sequence
    pc_seq = pc_seq[:,:, :2]
    final_pc_seq = pc_seq.reshape((len(time_step_indices) * total_points_per_timestep, 2))
    return final_pc_seq

def get_traj_pts(env, rel_pts, traj_mat, plot=False):
    """
    Returns the sequence of points representing the robot trajectory
    """
    traj_pts = np.zeros((len(traj_mat) * len(rel_pts), len(rel_pts[0])))
    for i in range(len(traj_mat)):
        transform_to_pc = traj_mat[i]
        rotation = transform_to_pc[:3,:3]
        translation = transform_to_pc[:,3]
        translation = translation[:3].reshape(3, 1)
        apply_t = lambda x: np.asarray(np.dot(rotation, x.reshape(3, 1)) + translation[:3]).reshape(-1)
        index = i * len(rel_pts)
        traj_pts[index: index + len(rel_pts)] = np.array(map(apply_t, rel_pts))
        if plot:
            plot_clouds(env, np.array(map(apply_t, rel_pts)))
    return traj_pts[:,:2]

def get_orig_pc_at_origin(rave_robot, initial_pc):
    """
    Returns the given point cloud centered at origin
    Make sure the origin pointcloud is centered at the transform though and not rotated!!
    ### correct ###
    """
    robot_t = rave_robot.GetTransform()
    trans = robot_t[:,3][:3]
    orig_pc_at_origin = np.zeros((len(initial_pc), len(initial_pc[0])))
    for i in range(len(initial_pc)):
        orig_pc_at_origin[i,:] = initial_pc[i] - trans
    return orig_pc_at_origin

def convert_traj_rel_pts_to_init_traj(traj_pts):
    """
    Only works for rectangular shape object
    """
    return

    assert len(traj_pts) % 4 == 0
    traj = np.zeros((len(traj_pts) // 4, 3))
    for i in range(len(traj_pts) // 4):
        index = 4 * i
        center1 = 0.5 * (traj_pts[index] + traj_pts[index + 3])
        center2 = 0.5 * (traj_pts[index + 1] + traj_pts[index + 2])
        assert center1 == center2
        # transform = np.matrixFromAxisAngle([0, 0, 
        # first and last are diagonal 
        # second and third are diagonal 

def get_center_points_from_rel_pts(traj_pts):
    assert len(traj_pts) % 4 == 0
    center_pts = np.zeros((len(traj_pts) // 4, len(traj_pts[0])))
    for i in range(len(traj_pts) // 4):
        index = 4 * i
        center = 0.5 * (traj_pts[index] + traj_pts[index + 3])
        center_pts[i,:] = center
    return center_pts
