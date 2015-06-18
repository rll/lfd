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

