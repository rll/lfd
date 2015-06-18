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
from lfd.registration.plotting_openrave import registration_plot_cb_2d
from lfd.collision_checker.collision_checker import CollisionChecker
from lfd.util.example_utils import * # (TODO) Don't import everything
from lfd.util.plotting_utils import *  # (TODO) Don't import everything

### Global variables
plot_grid=True

class FeedbackRegistrationAndTrajectoryTransferer(object):
    def __init__(self, env, 
                 registration_factory,
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
        self.registration_factory = registration_factory

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

    # @profile
    def transfer(self, demo, test_robot, test_scene_state, rel_pts, target_pose=None, timestep_dist=5, callback=None, plotting=False, include_timesteps = False):
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
        obstruction_pc = demo.obstruction_pc

        #### Plot demonstration robot trajectory ####
        if plotting:
            plot_robot(self.env, rel_pts_traj_seq)

        #### Get orig pc for use later ####
        if obstruction_pc == None:
            orig_pc_at_origin = get_orig_pc_at_origin(rave_robot, test_scene_state)
        else:
            length_of_orig_robot_pc = len(demo_pc_seq[0]) - len(obstruction_pc)
            orig_pc_at_origin = get_orig_pc_at_origin(rave_robot, test_scene_state[:length_of_orig_robot_pc])

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
        if not include_timesteps:
            lamb = np.zeros((total_pc_points, dim)) # dual variable for point cloud points 
        else:
            lamb = np.zeros((total_pc_points, dim + 1)) # dual variable for point cloud points 

        # convert the dimension of point cloud
        demo_pc = demo_pc_seq.reshape((total_pc_points, 3)) 
        if plotting:
            plot_sampled_demo_pc(self.env, demo_pc)
        demo_pc = demo_pc[:,:dim]
        time_steps_for_pc = None
        if include_timesteps:
            time_steps_for_pc = np.zeros(len(demo_pc))
            for i in range(len(time_step_indices)):
                index = i * points_per_pc
                time_steps_for_pc[index:index + points_per_pc] = np.ones(points_per_pc) * time_step_indices[i]
            time_steps_for_pc = time_steps_for_pc.reshape((len(time_steps_for_pc), 1))
            demo_pc = np.hstack((demo_pc, time_steps_for_pc))

        # convert trajectory to points 
        rel_pts_traj_seq = demo.rel_pts_traj_seq
        demo_traj_pts = rel_pts_traj_seq[:,:,:]  # (TODO) Implement the complicated version

        # dual variable for sequence of trajectories
        points_per_traj = len(demo_traj_pts[0])
        assert len(demo_traj_pts) == total_num_time_steps
        total_traj_points = points_per_traj * total_num_time_steps
        if not include_timesteps:
            nu_bd = np.zeros((total_traj_points, dim))
        else:
            nu_bd = np.zeros((total_traj_points, dim+1))
       
        # convert the dimension of tau_bd
        tau_bd = demo_traj_pts.reshape((total_traj_points, 3))
        tau_bd = tau_bd[:,:dim]
        time_steps_for_traj = None
        if include_timesteps:
            time_steps_for_traj = np.zeros(total_num_time_steps * points_per_traj)
            for i in range(total_num_time_steps):
                index = i * points_per_traj
                time_steps_for_traj[index:index + points_per_traj] = np.ones(points_per_traj) * i
            time_steps_for_traj = time_steps_for_traj.reshape((len(time_steps_for_traj), 1))
            tau_bd = np.hstack((tau_bd, time_steps_for_traj))

        # Set up Trajopt parameters
        # lin_traj_coeff = 10000.0 # (TODO) tune this
        # lin_pc_coeff = 10000.0
        lin_traj_coeff = 1000 # (TODO) tune this
        lin_pc_coeff = 1000
        penalty_coeffs = [7000]
        dist_pen = [0.005]
        nu_bd_trajopt_zeros = np.zeros((len(nu_bd), 1))
        lamb_trajopt_zeros = np.zeros((len(lamb), 1))

        #### traj diff threshold
        traj_diff_threshold = pow(10, -3) * nu_bd.size
        #### tps diff threhold
        pc_diff_threshold = pow(10, -3) * lamb.size
        #### maximum number of iterations
        max_iter = 20

        # set base link
        base_link = "base"

        ############## TPS #################
        # Setting parameters for tps 
        # bend_coef = 0.0001
        # rot_coef = np.array([0.0001, 0.0001])
        # bend_coef = 100000000 ## working
        bend_coef = 100000
        if not include_timesteps:
            rot_coef = np.array([0.001, 0.001])
        else:
            rot_coef = np.array([0.001, 0.001, 0.001])

        wt_n = None # unused for now

        #### initialize thin plate spline with the objective function C_TPS{f}
        first_demo_scene = demo_pc_seq[0]
        first_demo_scene = first_demo_scene[:,:2]
        test_scene_state = test_scene_state[:,:2]
        # if include_timesteps:
        #     first_demo_scene = np.hstack((first_demo_scene, np.zeros((len(first_demo_scene), 1))))
        #     test_scene_state = np.hstack((test_scene_state, np.zeros((len(test_scene_state), 1))))
        reg = self.registration_factory.register(first_demo_scene, test_scene_state, callback=None, using_feedback=True) # (TODO) Clean up code in this function
        f = reg.f

        ### initialize thin plate spline stuff
        if not include_timesteps:
            f_later = tps.ThinPlateSpline(dim)
        else:
            f_later = tps.ThinPlateSpline(dim+1)

        f_later.bend_coef = bend_coef
        f_later.rot_coef = rot_coef
        x_na = np.vstack((demo_pc, tau_bd))
        f_later.x_na = x_na
        # x_nd = np.hstack((x_na, np.ones((len(x_na), 1)))) 

        # Used for trajectory optimization
        curr_traj = mat_to_base_pose(demo_traj)
        # eta = 0.00000001
        if include_timesteps:
            eta = pow(10, -12)
        else:
            eta = pow(10, -14)
        # eta = pow(10, -14)
        # eta = pow(10, -12) # for the one with timesteps
        
        
        for iteration in range(max_iter):
            # lin_traj_coeff = lin_traj_coeff / 10
            # lin_pc_coeff = lin_pc_coeff / 10
            # lin_traj_coeff = lin_traj_coeff / 2
            # lin_pc_coeff = lin_pc_coeff / 2
            lin_traj_coeff = lin_traj_coeff / 1.5
            lin_pc_coeff = lin_pc_coeff / 1.5

            
            ########################################################
            ################### Set up trajopt #####################
            ########################################################

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
            # In the first iteration, initialize trajectory as demo trajectory. Later, initialize trajectory
            # as previous trajectory
            request["init_info"] = {
                "type": "given_traj",
                "data": curr_traj.tolist()
            }

            # penalty_coeffs = [3000]
            # dist_pen = [0.025]
            penalty_coeffs[0] = penalty_coeffs[0] * 1.05
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
                            "pos_coeffs":[lin_pc_coeff / num_pc_considered] * points_per_pc
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
            if iteration >= 1:
                ##### Compute difference in tps (pointcloud) #####
                orig_pc = demo.scene_states[0]
                new_pc_sampled = get_new_pc_from_traj(self.env, orig_pc_at_origin, time_step_indices, traj_mat, obstruction_pc = obstruction_pc, plot=False)
                if include_timesteps:
                    new_pc_sampled = np.hstack((new_pc_sampled, time_steps_for_pc))
                f_on_demo_pc_sampled = f.transform_points(demo_pc)
                assert new_pc_sampled.shape == f_on_demo_pc_sampled.shape
                pc_diff = new_pc_sampled - f_on_demo_pc_sampled
                if plotting:
                    draw_two_pc(self.env, new_pc_sampled, f_on_demo_pc_sampled, include_timesteps=include_timesteps)
                if not include_timesteps:
                    abs_pc_diff = sum(sum(abs(pc_diff)))
                else:
                    abs_pc_diff = sum(sum(abs(pc_diff[:,:2])))
                    
                print "Abs difference between sampled pointcloud: ", abs_pc_diff

     
                ##### Compute difference in trajectory #####
                # (TODO) figure out if use relative points or normal points here
                new_traj_pts = get_traj_pts(self.env, rel_pts, traj_mat, plot=False)
                if include_timesteps:
                    new_traj_pts = np.hstack((new_traj_pts, time_steps_for_traj))
                    
                f_on_demo_traj_rel_pts = f.transform_points(tau_bd) # (TODO)
                assert new_traj_pts.shape == f_on_demo_traj_rel_pts.shape
                traj_diff = new_traj_pts - f_on_demo_traj_rel_pts
                if plotting:
                    draw_two_traj(self.env, new_traj_pts, f_on_demo_traj_rel_pts, include_timesteps=include_timesteps)
                if not include_timesteps:
                    abs_traj_diff = sum(sum(abs(traj_diff)))
                else:
                    abs_traj_diff = sum(sum(abs(traj_diff[:,:2])))

                print "Abs difference between traj pts: ", abs_traj_diff
                
                # raw_input("inspect new variables and trajectories")


                ##### Break if the difference is under threshold ######
                if abs_traj_diff < traj_diff_threshold and abs_pc_diff < pc_diff_threshold:
                    break

                ###########################################
                ########## Update dual variables ##########
                ###########################################
                # substraction order looks correct
                eta = eta / 1.2
                lamb = lamb - eta * pc_diff 
                nu_bd = nu_bd - eta * traj_diff



            ########################################################
            ################### Solve tps problem ##################
            ########################################################
            ## TODO ##
            theta = tps.tps_fit_feedback(demo_pc, None, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)
            print("============ theta ============")
            print(theta)
            f = f_later
            f.update_theta(theta)
            
            # if plot_grid:
            #     y_md = np.vstack((new_pc_sampled, new_traj_pts))
            #     registration_plot_cb_2d(self.env.sim, x_na, y_md, f, z)
            
            ######## get transfered trajectory from current f ######## 
            # raw_input("aalling transform points")
            f_on_demo_traj_rel_pts = f.transform_points(tau_bd) # (TODO)
            if not include_timesteps:
                f_on_demo_traj_rel_pts = np.hstack((f_on_demo_traj_rel_pts, np.zeros((len(f_on_demo_traj_rel_pts), 1))))
            else:
                f_on_demo_traj_rel_pts[:,2] = np.zeros(len(f_on_demo_traj_rel_pts))

            if plotting:
                plot_robot(self.env, f_on_demo_traj_rel_pts, "look at f(tau_bd) after solving for new f")

        ### Check if final trajectory is collision free
        collision_checker = CollisionChecker(self.sim.env)
        bodypart_traj = {}
        bodypart_traj['base'] = curr_traj
        collisions = collision_checker.get_traj_collisions(bodypart_traj, include_obj_world_col=False)
        if collisions == set():
            raw_input("Final trajectory is collision free")
        else:
            raw_input("Final trajectory is not collision free: {}".format(collisions))

        return traj_mat
