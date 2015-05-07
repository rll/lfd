import copy
import datetime
import settings
import json
import trajoptpy
import openravepy
import numpy as np
import sys
from lfd.demonstration import demonstration
from lfd.environment import sim_util
from lfd.registration import registration, tps
from lfd.transfer import transfer
from lfd.transfer import planning
from lfd.util import util

def base_pose_to_mat(traj, z):
    result = np.zeros((len(traj), 4, 4))
    for i in range(len(traj)):
        pose = traj[i]
        x, y, rot = pose
        q = openravepy.quatFromAxisAngle((0, 0, rot)).tolist()
        pos = [x, y, z]
        mat = openravepy.matrixFromPose(q + pos)
        # import pdb; pdb.set_trace()
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

def plot_traj(env, traj):
    """ 
    param env - openrave environment
    param traj - a list of pose (4x4 matrix)
    """
    pass

def plot_clouds(env, pc_seqs):
    # for pc in pc_seqs:
    handles = []
    handles.append(env.sim.env.plot3(points = pc_seqs, pointsize=3, colors=[0, 1, 0], drawstyle=1))
    env.sim.viewer.Step()
    raw_input("look at pc")

def get_new_pc_from_traj(env, orig_pc_at_origin, time_step_indices, traj_mat, plot=False):
    """
    Given a trajectory, returns the pointcloud sequence 
    """
    traj = traj_mat[time_step_indices] ## timesteps that matters
    pc_seq = np.zeros(((len(traj)), orig_pc_at_origin.shape[0], 3))
    for i in range(len(traj)):
        transform_to_pc = traj[i]
        rotation = transform_to_pc[:3,:3]
        translation = transform_to_pc[:,3]
        translation = translation[:3].reshape(3, 1)
        apply_t = lambda x: np.asarray(np.dot(rotation, x.reshape(3, 1)) + translation[:3]).reshape(-1)
        pc_seq[i,:,:] = np.array(map(apply_t, orig_pc_at_origin))
        if plot:
            plot_clouds(env, np.array(map(apply_t, orig_pc_at_origin)))

    # returns a two dimension pointcloud sequence
    pc_seq = pc_seq[:,:, :2]
    final_pc_seq = pc_seq.reshape((len(time_step_indices) * len(orig_pc_at_origin), 2))
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

class FeedbackRegistrationAndTrajectoryTransferer(object):
    def __init__(self, env, 
                 alpha=settings.ALPHA, # alpha not used.
                 beta_pos=settings.BETA_POS,
                 gamma=settings.GAMMA,
                 use_collision_cost=settings.USE_COLLISION_COST):

        self.sim = env.sim
        self.env = env
        self.alpha = alpha
        self.beta_pos = beta_pos
        self.gamma = gamma # joint velocity constant
        self.use_collision_cost = use_collision_cost

    def traj_to_points(self, traj):
        """Convert trajectory to points"""
        pass

    def get_scene_state(self, robot, num_x_points, num_y_points):
        """
        Get scene state (a set of points)  given the robot 
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
        return init_pc

    def traj_to_points(self, robot, traj, num_x_points = 3 , num_y_points = 12, plot=False):
        """
        Convert trajectory to points
        """
        # sample points from the robot (initial pc)
        robot_kinbody = robot.get_bullet_objects()[0].GetKinBody()
        init_pc = self.get_scene_state(robot, num_x_points, num_y_points)

        init_t = robot_kinbody.GetTransform()
        y_points = num_y_points
        x_points = num_x_points
        total_points = y_points * x_points
        min_x, max_x, min_y, max_y, z = get_object_limits(robot_kinbody)

        # generate pc from trajectory
        pc_seq = np.empty(((len(traj)), total_points, 3))
        pc_seq[0,:,:] = init_pc
        center_pt = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, z]).reshape(3, 1)
        for i in range(1, len(traj)):
            transform_to_pc = traj[i-1]
            rotation = transform_to_pc[:3,:3]
            translation = transform_to_pc[:,3] - init_t[:,3]
            translation = translation[:3].reshape(3, 1)
            apply_t = lambda x: np.asarray((np.dot(rotation, x.reshape(3, 1) - center_pt)) + center_pt + translation[:3]).reshape(-1)
            pc_seq[i,:,:] = np.array(map(apply_t, init_pc))
        return pc_seq

        
    def points_to_array(self, pts_traj):
        """ Convert points to a flattened numpy array, where each element is an array of length 3"""
        tmp = pts_traj 
        return tmp.reshape(tmp.shape[0] * tmp.shape[1], tmp.shape[2])

    def compact_traj(self, traj):
        """
        Compactly represent the list of pose as a list of three-array list (x, y, theta)
        (Not needed for now)
        """
        z = traj[2, 3]
        x, y = traj[:2, 3]
        theta = None# arctan
        return (x, y, theta)

    # @profile
    def transfer(self, demo, test_robot, test_scene_state, rel_pts, target_pose=None, timestep_dist=5, callback=None, plotting=False):
        """
        Trajectory transfer of demonstrations using dual decomposition incorporating feedback
        """
        ### TODO: Need to tune parameters !!
        # print 'alpha = ', self.alpha
        # print 'beta = ', self.beta_pos
        print 'gamma = ', self.gamma # only gamma in use (for cost of trajectory)
        dim = 2 

        # Get openrave robot
        rave_robot = self.env.sim.env.GetRobots()[0]
        z = rave_robot.GetTransform()[2, 3]

        ######### Initialization #########
        demo_pc_seq = demo.scene_states
        demo_traj = demo.traj
        rel_pts_traj_seq = demo.rel_pts_traj_seq

        #### Get orig pc for use later ####
        orig_pc_at_origin = get_orig_pc_at_origin(rave_robot, demo_pc_seq[0])

        ### Comment out when no target pose is specified
        target_pose_7 = openravepy.poseFromMatrix(target_pose).tolist()

        # Determines how many time steps of all the demos to actually use (0, timestep_dist, 2*timestep_dist, ...)
        total_num_time_steps = len(demo_pc_seq)
        assert type(timestep_dist) == int
        time_step_indices = np.arange(0, total_num_time_steps, timestep_dist)
        demo_pc_seq = demo_pc_seq[time_step_indices] ## The new demo_pc_seq

        # dual variable for sequence of point clouds
        points_per_pc = len(demo_pc_seq[0])
        pc_num_time_steps = len(demo_pc_seq)
        total_pc_points = points_per_pc * pc_num_time_steps
        lamb = np.zeros((total_pc_points, dim)) # dual variable for point cloud points 

        # convert the dimension of point cloud
        demo_pc = demo_pc_seq.reshape((total_pc_points, 3)) 
        demo_pc = demo_pc[:,:dim]

        # convert trajectory to points 
        rel_pts_traj_seq = demo.rel_pts_traj_seq
        demo_traj_pts = rel_pts_traj_seq[:,:,:]  # (TODO) Implement the complicated version

        # dual variable for sequence of trajectories
        points_per_traj = len(demo_traj_pts[0])
        assert len(demo_traj_pts) == total_num_time_steps
        total_traj_points = points_per_traj * total_num_time_steps
        nu_bd = np.zeros((total_traj_points, dim))
       
        # convert the dimension of tau_bd
        tau_bd = demo_traj_pts.reshape((total_traj_points, 3))
        tau_bd = tau_bd[:,:dim]

        # Set up Trajopt parameters
        # lin_traj_coeff = 10000.0 # (TODO) tune this
        # lin_pc_coeff = 10000.0
        lin_traj_coeff = 1000.0 # (TODO) tune this
        lin_pc_coeff = 1000.0
        # penalty_coeffs = [7000]
        # penalty_coeffs = [7000]
        # dist_pen = [0.020]
        penalty_coeffs = [10000]
        dist_pen = [0.020]
        nu_bd_trajopt_zeros = np.zeros((len(nu_bd), 1))
        lamb_trajopt_zeros = np.zeros((len(lamb), 1))

        #### traj diff threshold
        traj_diff_threshold = pow(10, -3) * nu_bd.size
        #### tps diff threhold
        pc_diff_threshold = pow(10, -3) * lamb.size
        #### maximum number of iterations
        max_iter = 5


        ############## TPS #################
        # Setting parameters for tps 
        bend_coef = 0.0001
        rot_coef = np.array([0.0001, 0.0001])
        wt_n = None # unused for now
        # theta = tps.tps_fit_feedback(demo_pc, None, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)

        ### initialize thin plate spline stuff
        f = tps.ThinPlateSpline(dim)
        f.bend_coef = bend_coef
        f.rot_coef = rot_coef
        # f.theta = theta
        x_na = np.vstack((demo_pc, tau_bd))
        base_link = "base"
        theta = tps.tps_fit_feedback(demo_pc, None, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)
        import pdb; pdb.set_trace()
        f.x_na = x_na
        f.update_theta(theta)

        ######## Initialize trajectory with current f ###########   
        #### what am i doing??
        # f_on_demo_traj_rel_pts = f.transform_points(tau_bd)# (TODO)
        # center_pts = get_center_points_from_rel_pts(f_on_demo_traj_rel_pts)
        # import pdb; pdb.set_trace()
        # pc = np.hstack((center_pts, np.zeros((len(center_pts), 1))))
        # raw_input("initial: look at f(tau_bd)")
        # plot_clouds(self.env, pc)
        curr_traj = mat_to_base_pose(demo_traj)
        
        
        for i in range(max_iter):
            ########################################################
            ################### Set up tps #########################
            ########################################################
            ## TODO ##
            # theta = tps.tps_fit_feedback(demo_pc, None, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)
            # f.theta = theta
            
            ########################################################
            ################### Set up trajopt #####################
            ########################################################

            ######## Initialize trajectory with current f ##########
            f_on_demo_traj_rel_pts = f.transform_points(tau_bd) # (TODO)
            center_pts = get_center_points_from_rel_pts(f_on_demo_traj_rel_pts)
            # pc = np.hstack((f_on_demo_traj_rel_pts, np.zeros((len(f_on_demo_traj_rel_pts), 1))))
            pc = np.hstack((center_pts, np.zeros((len(center_pts), 1))))
            plot_clouds(self.env, pc)

            costs = []
            constraints = []

            # joint velocity cost
            joint_vel_cost = {
                "type": "joint_vel",
                "params": {"coeffs": [self.gamma/(total_num_time_steps - 1)]}
            }
            costs.append(joint_vel_cost)

            ### Set target pose (comment out later)
            # constraints.append({
            #     "type": "pose",
            #     "params": {"xyz": target_pose_7[4:],
            #                "wxyz": target_pose_7[:4],
            #                "link": base_link,
            #                "pos_coeffs": [20, 20, 20],
            #                "rot_coeffs": [20, 20, 20]}
            # })

            request = {
                "basic_info": {
                  "n_steps": total_num_time_steps,
                  "manip": "base",
                  "start_fixed": False  # i.e., DOF values at first timestep are fixed based on current robot state
                },
                "costs": costs,
                "constraints": constraints,
                "init_info":  {
                    "type": "stationary"
                }
            }
            request["init_info"] = {
                "type": "given_traj",
                "data": curr_traj.tolist()
            }

            # penalty_coeffs = [3000]
            # dist_pen = [0.025]
            request['costs'] += [{
                "type": "collision",
                "name": "col",
                "params": {
                  "continuous": True,
                  "coeffs": penalty_coeffs,
                  "dist_pen": dist_pen
                }
            }]

            ######### Adding linear term for trajectory relative points cost ######## 
            #### Should consider all the time steps #####
            assert points_per_traj == len(rel_pts) 
            nu_bd_trajopt = np.hstack((nu_bd, nu_bd_trajopt_zeros))
            for i in range(total_num_time_steps):
                request['costs'].append(
                    {"type":"rel_pts_nus",
                        "params":{
                            "nus": (-nu_bd_trajopt[i * points_per_traj: (i+1) * points_per_traj]).tolist(),
                            "rel_xyzs": rel_pts.tolist(),
                            "link": base_link,
                            "timestep": i,
                            "pos_coeffs":[lin_traj_coeff / total_num_time_steps] * 4
                        }
                    }
                )

            ######### Adding linear term for sampled pointcloud cost ######## 
            ### (TODO) cost needs to be reconsidered 
            num_pc_considered = len(time_step_indices)
            lamb_trajopt = np.hstack((lamb, lamb_trajopt_zeros))
            for i in range(num_pc_considered):
                timestep = time_step_indices[i]
                request['costs'].append(
                    {"type": "pc_pts_lambdas",
                        "params": {
                            "lambdas": (-lamb_trajopt[i * points_per_pc: (i+1) * points_per_pc]).tolist(),
                            "orig_pc": orig_pc_at_origin.tolist(),
                            "link": base_link,
                            "timestep": timestep,
                            # "num_pc_considered": num_pc_considered,
                            # "pc_time_steps": time_step_indices,
                            "pos_coeffs":[lin_pc_coeff / num_pc_considered] * 4
                        }
                    }
                )

            ####### Construct and solve trajopt problem ########
            with openravepy.RobotStateSaver(rave_robot):
                prob = trajoptpy.ConstructProblem(json.dumps(request), self.env.sim.env)
                result = trajoptpy.OptimizeProblem(prob)
            traj = result.GetTraj()
            total_cost = sum(cost[1] for cost in result.GetCosts())
            traj_mat = base_pose_to_mat(traj, z)
            curr_traj = traj ## variable cur_traj used in initializing trajopt
            
            ## executes the trajectory of the robot
            # self.env.execute_robot_trajectory(rave_robot, traj_mat)

            ######
            # plot_traj(env.sim.env, traj_mat)

            ##### Compute difference in tps (pointcloud) #####
            orig_pc = demo.scene_states[0]
            new_pc_sampled = get_new_pc_from_traj(self.env, orig_pc_at_origin, time_step_indices, traj_mat, plot=False)
            f_on_demo_pc_sampled = f.transform_points(demo_pc)
            assert new_pc_sampled.shape == f_on_demo_pc_sampled.shape
            pc_diff = new_pc_sampled - f_on_demo_pc_sampled
            abs_pc_diff = sum(sum(abs(pc_diff)))
            print "Abs difference between sampled pointcloud: ", abs_pc_diff

 
            ##### Compute difference in trajectory #####
            # (TODO) figure out if use relative points or normal points here
            new_traj_pts = get_traj_pts(self.env, rel_pts, traj_mat, plot=True)
            f_on_demo_traj_rel_pts = f.transform_points(tau_bd) # (TODO)
            assert new_traj_pts.shape == f_on_demo_traj_rel_pts.shape
            traj_diff = new_traj_pts - f_on_demo_traj_rel_pts
            abs_traj_diff = sum(sum(abs(traj_diff)))
            print "Abs difference between traj pts: ", abs_traj_diff
            
            raw_input("inspect new variables and trajectories")
            import pdb; pdb.set_trace()


            ##### Compute cost and print ######
            if abs_traj_diff < traj_diff_threshold and abs_pc_diff < pc_diff_threshold:
                break

            ###########################################
            ########## Update dual variables ##########
            ###########################################
            eta = 0.0001
            lamb = lamb - eta * pc_diff
            nu_bd = nu_bd - eta * traj_diff
            # lamb = lamb + eta * pc_diff
            # nu_bd = nu_bd + eta * traj_diff

            ########################################################
            ################### Set up tps #########################
            ########################################################
            ## TODO ##
            theta = tps.tps_fit_feedback(demo_pc, None, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)
            f.update_theta(theta)
            
            ######## get transfered trajectory from current f ######## 
            f_on_demo_traj_rel_pts = f.transform_points(tau_bd) # (TODO)
            center_pts = get_center_points_from_rel_pts(f_on_demo_traj_rel_pts)
            # init_traj = convert_traj_rel_pts_to_init_traj(f_on_demo_traj_rel_pts)
            # pc = np.hstack((f_on_demo_traj_rel_pts, np.zeros((len(f_on_demo_traj_rel_pts), 1))))
            pc = np.hstack((center_pts, np.zeros((len(center_pts), 1))))
            raw_input("look at f(tau_bd)")
            plot_clouds(self.env, pc)


        return traj_mat

