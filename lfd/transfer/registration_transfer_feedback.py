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

class RegistrationAndTrajectoryTransferer(object):
    def __init__(self, registration_factory, trajectory_transferer):
        self.registration_factory = registration_factory
        self.trajectory_transferer = trajectory_transferer

    def transfer(self, demo, test_scene_state, callback=None, plotting=False):
        """Registers demonstration scene onto the test scene and uses this registration to transfer the demonstration trajectory

        Args:
            demos: A list of Demonstration that has the demonstration scene and the trajectory to transfer
            test_scene_state: SceneState of the test scene

        Returns:
            The transferred Trajectory
        """
        raise NotImplementedError

class FeedbackRegistrationAndTrajectoryTransferer(RegistrationAndTrajectoryTransferer):
    def __init__(self, registration_factory, trajectory_transferer,
                 alpha=settings.ALPHA, # alpha not used.
                 beta_pos=settings.BETA_POS,
                 gamma=settings.GAMMA,
                 use_collision_cost=settings.USE_COLLISION_COST,
                 init_trajectory_transferer=None):
        super(FeedbackRegistrationAndTrajectoryTransferer, self).__init__(registration_factory, trajectory_transferer)
        if not isinstance(registration_factory, registration.TpsRpmRegistrationFactory):
            raise NotImplementedError("FeedbackRegistrationAndTrajectoryTransferer only supports TpsRpmRegistrationFactory")
        if not isinstance(trajectory_transferer, transfer.FingerTrajectoryTransferer):
            raise NotImplementedError("FeedbackRegistrationAndTrajectoryTransferer only supports FingerTrajectoryTransferer")
        self.sim = trajectory_transferer.sim
        self.alpha = alpha
        self.beta_pos = beta_pos
        self.gamma = gamma # joint velocity constant
        self.use_collision_cost = use_collision_cost
        self.init_trajectory_transferer = init_trajectory_transferer

    def traj_to_points(self, aug_traj, resampling=False):
        """Convert augmented trajectory trajectory to points"""
        active_lr = "r"

        # rs resampling
        if resampling:
            _, timesteps_rs = sim_util.unif_resample(np.c_[(1./settings.JOINT_LENGTH_PER_STEP) * np.concatenate([aug_traj.lr2arm_traj[lr] for lr in active_lr], axis=1),
                                                           (1./settings.FINGER_CLOSE_RATE) * np.concatenate([aug_traj.lr2finger_traj[lr] for lr in active_lr], axis=1)],
                                                     1.)
            demo_aug_traj_rs = aug_traj.get_resampled_traj(timesteps_rs)
        else:
            demo_aug_traj_rs = aug_traj

        lr = 'r'
        arm_name = {"rl":"leftarm", "r":"rightarm"}[lr]
        finger_name = "%s_gripper_l_finger_joint"%lr

        flr2demo_finger_pts_traj_rs = sim_util.get_finger_pts_traj(self.sim.robot, lr, (demo_aug_traj_rs.lr2ee_traj[lr], demo_aug_traj_rs.lr2finger_traj[lr]))
        return flr2demo_finger_pts_traj_rs


    def points_to_array(self, flr2demo_finger_pts_traj):
        """ Convert points to a flattened numpy array, where each element is an array of length 3"""
        temp =  np.r_[flr2demo_finger_pts_traj['l'], flr2demo_finger_pts_traj['r']]
        return temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2])

    # @profile
    def transfer(self, demos, test_scene_state, callback=None, plotting=False):
        """
        Trajectory transfer of demonstrations with two segments (baby version)
        """
        print 'alpha = ', self.alpha
        print 'beta = ', self.beta_pos
        print 'gamma = ', self.gamma
        demo1 = demos[0]
        demo2 = demos[1]
        reg = self.registration_factory.register(demo1, test_scene_state, callback=callback)

        ######## INITIALIZATION ##########

        # Demonstration trajectory points in an array
        tau_bd1 = self.points_to_array(self.traj_to_points(demo1.aug_traj, resampling=True))
        tau_bd2 = self.points_to_array(self.traj_to_points(demo2.aug_traj, resampling=True))

        # Dual variables for the trajectories
        nu_bd1 = np.zeros(tau_bd1.shape) # 456 x 3
        nu_bd2 = np.zeros(tau_bd2.shape) 

        # TPS Parameters, point clouds, etc. (what the heck is this?)
        (n,d) = reg.f.x_na.shape # 260 x 3

        # bend coefficientsm, reg.f.bend_coef = 0.0001
        bend_coefs = np.ones(d) * reg.f.bend_coef if np.isscalar(reg.f.bend_coef) else reg.f.bend_coef
        # rotation coefficients, reg.f.rot_coef = [0.0001, 0.0001, 0.1]
        rot_coefs = np.ones(d) * reg.f.rot_coef if np.isscalar(reg.f.rot_coef) else reg.f.rot_coef

        # what does these three functions mean
        x_na = reg.f.x_na # matrix of size 260x3
        y_ng = reg.f.y_ng # matrix of size 260x3
        wt_n = reg.f.wt_n # vector of length 260

        # apply registration f on the demo trajectory points
        warped_points = reg.f.transform_points(tau_bd1) # 456 x 3

        # construct target_traj and finger points for the new trajectory
        target_traj = []
        i = 0
        finger_points = []
        for point in warped_points:
            finger_points.append(point)
            # why every four poins?
            if i % 4 == 3:
                target_traj.append(np.array(finger_points))
                finger_points = []
            i = i+1

        # plot demo cloud, test scene cloud, demo traj, new traj
        handles = []
        lr = 'r'
        if plotting:
            demo_cloud = demo1.scene_state.cloud
            test_cloud = reg.test_scene_state.cloud
            demo_color = demo1.scene_state.color
            test_color = reg.test_scene_state.color
            handles.append(self.sim.env.plot3(demo_cloud[:,:3], 2, demo_color if demo_color is not None else (1,0,0)))
            handles.append(self.sim.env.drawlinestrip(demo1.aug_traj.lr2ee_traj[lr][:,:3,3], 2, (1,0,0))) # shows demo trajectory
            handles.extend(sim_util.draw_finger_pts_traj(self.sim, {'r':target_traj}, (0,0,1))) # shows target trajectory
            if self.sim.viewer:
              self.sim.viewer.Step()

        # get active manipulator
        active_lr = ""
        for lr in 'lr':
            if lr in demo1.aug_traj.lr2arm_traj and sim_util.arm_moved(demo1.aug_traj.lr2arm_traj[lr]):
                active_lr += lr

        # create demo augmented trajectory rs for active manipulator (this has been done before if resampling == True)
        # same as the function traj to points
        _, timesteps_rs = sim_util.unif_resample(np.c_[(1./settings.JOINT_LENGTH_PER_STEP) * np.concatenate([demo1.aug_traj.lr2arm_traj[lr] for lr in active_lr], axis=1),
                                                       (1./settings.FINGER_CLOSE_RATE) * np.concatenate([demo1.aug_traj.lr2finger_traj[lr] for lr in active_lr], axis=1)],
                                                 1.)
        demo_aug_traj_rs = demo1.aug_traj.get_resampled_traj(timesteps_rs) # still an AugmentedTrajectory object


        ## 
        manip_name = ""
        flr2finger_link_names = [] # left/right finger link names
        flr2demo_finger_pts_trajs_rs = [] # left/right finger points trajectory 

        init_traj = np.zeros((len(timesteps_rs),0)) # initial trajectory
        for lr in active_lr:
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            finger_name = "%s_gripper_l_finger_joint"%lr

            if manip_name:
                manip_name += "+"
            manip_name += arm_name + "+" + finger_name
            
            # update init traj 
            # demo_aug_traj_rs.lr2arm_traj[lr]: 57x7
            # demo_aug_traj_rs.lr2finger_traj[lr]: 1x7
            # init_traj 
            init_traj = np.c_[init_traj, demo_aug_traj_rs.lr2arm_traj[lr], demo_aug_traj_rs.lr2finger_traj[lr]]

            if plotting:
                # difference between resampling and no resampling?
                handles.append(self.sim.env.drawlinestrip(demo1.aug_traj.lr2ee_traj[lr][:,:3,3], 2, (1,0,0)))
                handles.append(self.sim.env.drawlinestrip(demo_aug_traj_rs.lr2ee_traj[lr][:,:3,3], 2, (1,1,0)))
                transformed_ee_traj_rs = reg.f.transform_hmats(demo_aug_traj_rs.lr2ee_traj[lr]) # hmats
                handles.append(self.sim.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                if self.sim.viewer:
                  self.sim.viewer.Step()

            # left / right finger points trajectory, what does the below function do?
            flr2demo_finger_pts_traj_rs = sim_util.get_finger_pts_traj(self.sim.robot, lr, (demo_aug_traj_rs.lr2ee_traj[lr], demo_aug_traj_rs.lr2finger_traj[lr]))
            flr2demo_finger_pts_trajs_rs.append(flr2demo_finger_pts_traj_rs) # before the first for loop: flr2demo_finger_pts_trajs_rs
            
            flr2transformed_finger_pts_traj_rs = {} # dictionary containing transformed finger points trajectory
            flr2finger_link_name = {}
            flr2finger_rel_pts = {} # relative points

            for finger_lr in 'lr':
                flr2transformed_finger_pts_traj_rs[finger_lr] = reg.f.transform_points(np.concatenate(flr2demo_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                flr2finger_link_name[finger_lr] = "%s_gripper_%s_finger_tip_link"%(lr,finger_lr)
                flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)

            flr2finger_link_names.append(flr2finger_link_name) # this was originally: flr2finger_link_names = []


            # why do we need to plot so many times?
            if plotting:
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2demo_finger_pts_traj_rs, (1,1,0)))
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                if self.sim.viewer:
                  self.sim.viewer.Step()

        # Out of the for loop
        if not self.init_trajectory_transferer:
            # modify the shoulder joint angle of init_traj to be the limit (highest arm) because this usually gives a better local optima (but this might not be the right thing to do)
            dof_inds = sim_util.dof_inds_from_name(self.sim.robot, manip_name)
            joint_ind = self.sim.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
            init_traj[:,dof_inds.index(joint_ind)] = self.sim.robot.GetDOFLimits([joint_ind])[0][0]

        print "planning joint TPS and finger points trajectory following"
        robot = self.sim.robot
        flr2demo_finger_pts_trajs = flr2demo_finger_pts_trajs_rs # okay.....
        f = reg.f

        n_steps = init_traj.shape[0] # init_traj.shape: (57, 8)
        dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
        assert init_traj.shape[1] == len(dof_inds) # make sure that the init_traj dofs is the same as dofs for manip name
        # flr2demo_finger_pts_trajs: a list of dictionaries, where each dictionary contains keys 'r', 'l' and values of length n_steps
        # this one has length one
        for flr2demo_finger_pts_traj in flr2demo_finger_pts_trajs:
            for demo_finger_pts_traj in flr2demo_finger_pts_traj.values():
                assert len(demo_finger_pts_traj)== n_steps
        assert len(flr2finger_link_names) == len(flr2demo_finger_pts_trajs)

        # expand these
        (n,d) = f.x_na.shape
        if f.wt_n is None:
            wt_n = np.ones(n)
        else:
            wt_n = f.wt_n
        if wt_n.ndim == 1:
            wt_n = wt_n[:,None]
        if wt_n.shape[1] == 1:
            wt_n = np.tile(wt_n, (1,d))

        N = f.N # (264, 260)
        init_z = f.z # (260, 3)

        start_fixed = False
        if start_fixed:
            init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]] # start at where it is now
            sim_util.unwrap_in_place(init_traj, dof_inds)
            init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]
            
        # trajopt request
        request = {
            "basic_info" : {
                "n_steps" : n_steps,
                "manip" : manip_name,
                "start_fixed" : start_fixed
            },
            "costs" : [
            {
                "type" : "joint_vel",
                "params": {"coeffs" : [self.gamma/(n_steps-1)]}
            },
            ],
            "constraints" : [
            ],
        }

        # Add collision cost
        penalty_coeffs = [3000]
        dist_pen = [0.025]
        if self.use_collision_cost:
            request["costs"].append(
                {
                    "type" : "collision",
                    "params" : {
                      "continuous" : True,
                      "coeffs" : penalty_coeffs,  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                      "dist_pen" : dist_pen  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                    }
                })

        # Now that we've made the initial request that is the same every iteration,
        # we make the loop and add on the things that change.
        eta = 0.0001 # step size for updating nu_bd, trajectory threshold
        traj_diff_thresh = 1e-3 * nu_bd.size
        max_iter = 10
        cur_traj = init_traj
        del handles

        ########### MAIN LOOP ###############
        for itr in range(max_iter):
          handles = []
          request_i = copy.deepcopy(request)
          flr2transformed_finger_pts_traj = {}
          # right arm only.
          for finger_lr in 'lr':
            # what is this doing?
            flr2transformed_finger_pts_traj[finger_lr] = f.transform_points(np.concatenate(flr2demo_finger_pts_trajs[0][finger_lr], axis=0)).reshape((-1,4,3))
          flr2transformed_finger_pts_trajs = [flr2transformed_finger_pts_traj]

          # added current trajectory to the request to trajopt
          request_i["init_info"] = {
                "type":"given_traj",
                "data":[x.tolist() for x in cur_traj],
            }

          # Add nus to the trajectory optimization problem (the dual term variable)
          traj_dim = int(nu_bd.shape[0] / 2) # nu_bd.shape: (456, 3) # first half of nu_bd is for the left arm
          for i_step in range(0, traj_dim*2, 4):
            # handle left and right arm
            if i_step < traj_dim:
               finger_lr = 'l'
               traj_step = i_step
            else:
              finger_lr = 'r'
              traj_step = i_step - traj_dim
                
            # get the finger link name and rel_pts
            finger_link_name = flr2finger_link_name[finger_lr]
            finger_rel_pts = flr2finger_rel_pts[finger_lr]

            if start_fixed and traj_step==0: continue
            request_i["costs"].append(
                 {"type":"rel_pts_lambdas",
                   "params":{
                     # "lambdas":(-nu_bd[traj_step:traj_step+4,:]).tolist(),
                     "lambdas":(-nu_bd[traj_step:traj_step+4,:]).tolist(),
                     "rel_xyzs":finger_rel_pts.tolist(),
                     "link":finger_link_name,
                     "timestep":traj_step/4,
                     "pos_coeffs":[self.beta_pos/n_steps]*4,
                     }
            })
            
          # hmmmm. ok, used for coefficients 
          if itr == 1:
            beta_pos = self.beta_pos / 30
          elif itr < 3:
            beta_pos = self.beta_pos / 3
          else:
            beta_pos = self.beta_pos

          # what does this cost deal with?
          for (flr2finger_link_name, flr2transformed_finger_pts_traj) in zip(flr2finger_link_names, flr2transformed_finger_pts_trajs):
              for finger_lr, finger_link_name in flr2finger_link_name.items():
                  finger_rel_pts = flr2finger_rel_pts[finger_lr]
                  transformed_finger_pts_traj = flr2transformed_finger_pts_traj[finger_lr]
                  for (i_step, finger_pts) in enumerate(transformed_finger_pts_traj):
                      if start_fixed and i_step == 0:
                          continue
                      request_i["costs"].append(
                          {"type":"rel_pts",
                          "params":{
                              "xyzs":finger_pts.tolist(),
                              "rel_xyzs":finger_rel_pts.tolist(),
                              "link":finger_link_name,
                              "timestep":i_step,
                              "pos_coeffs":[np.sqrt(beta_pos/n_steps)]*4,
                            }
                          })

          # solve the optimization problem 
          s_traj = json.dumps(request_i);
          with openravepy.RobotStateSaver(robot):
            with util.suppress_stdout():
                prob = trajoptpy.ConstructProblem(s_traj, robot.GetEnv())
                if plotting:
                  viewer = trajoptpy.GetViewer(robot.GetEnv())
                  trajoptpy.SetInteractive(True)
            if itr == 0:
              result = trajoptpy.OptimizePartialProblem(prob, 5)
            else:
              result = trajoptpy.OptimizePartialProblem(prob, 2)
          cur_traj = result.GetTraj() # from trajectory optimization in the first step

          # Compute difference between trajectory points.
          trajpts_tps = f.transform_points(tau_bd)

          # Below is probably the same as doing:
          full_traj = (cur_traj, sim_util.dof_inds_from_name(self.sim.robot, manip_name))
          trajpts_traj = self.points_to_array(sim_util.get_finger_pts_traj(self.sim.robot, 'r', full_traj))

          #trajpts_traj = self.points_to_array(self.traj_to_points(self.opttraj_to_augtraj(cur_traj, manip_name, demo_aug_traj_rs.lr2open_finger_traj, demo_aug_traj_rs.lr2close_finger_traj),resampling=True))
          traj_diff = trajpts_traj - trajpts_tps; # f(tau_d) - tau
          abs_traj_diff = sum(sum(abs(traj_diff)))

          obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
          print "Absolute diff between traj pts: ", abs_traj_diff, ". Warp cost: ", f.get_objective()
          print "obj_value ", obj_value

          # update values of coefficient nu
          nu_bd = nu_bd - eta * traj_diff # eta = 0.0001,  nu - nu - eta(f(tau_d) - f(tau)

          # update f, look at how this is updated
          theta, (N, z) = tps.tps_fit_decomp(x_na, y_ng, bend_coefs, rot_coefs, wt_n, tau_bd, -nu_bd, ret_factorization=True)
          f.update(x_na, y_ng, bend_coefs, rot_coefs, wt_n, theta, N=N, z=z)

          lr = 'r'

          if plotting:
              handles.extend(sim_util.draw_finger_pts_traj(self.sim, {'r':trajpts_tps.reshape((-1,4,3))}, (0,1,0)))
              handles.extend(sim_util.draw_finger_pts_traj(self.sim, {'r':trajpts_traj.reshape((-1, 4,3))}, (1,0,0)))
              if self.sim.viewer:
                self.sim.viewer.Step()
                self.sim.viewer.Idle()

          # Trajectory has converged
          if abs_traj_diff < traj_diff_thresh:
            break


        # outside of the for loop iteration
        full_traj = (cur_traj, sim_util.dof_inds_from_name(self.sim.robot, manip_name))
        test_aug_traj = demonstration.AugmentedTrajectory.create_from_full_traj(self.sim.robot, full_traj, lr2open_finger_traj=demo_aug_traj_rs.lr2open_finger_traj, lr2close_finger_traj=demo_aug_traj_rs.lr2close_finger_traj)

        handles = []
        trajpts_traj = self.points_to_array(sim_util.get_finger_pts_traj(self.sim.robot, 'r', full_traj))
        if plotting:
            handles.extend(sim_util.draw_finger_pts_traj(self.sim, {'r':trajpts_traj.reshape((-1, 4,3))}, (1,0,0)))
            for lr in active_lr:
                flr2new_transformed_finger_pts_traj_rs = {}
            if self.sim.viewer:
              self.sim.viewer.Step()
              self.sim.viewer.Idle()

        return test_aug_traj
