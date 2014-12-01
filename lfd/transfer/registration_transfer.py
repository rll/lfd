from __future__ import division

import settings
import numpy as np
from lfd.demonstration import demonstration
from lfd.environment import sim_util
from lfd.registration import registration
from lfd.transfer import transfer
from lfd.transfer import planning

class RegistrationAndTrajectoryTransferer(object):
    def __init__(self, registration_factory, trajectory_transferer):
        self.registration_factory = registration_factory
        self.trajectory_transferer = trajectory_transferer

    def transfer(self, demo, test_scene_state, callback=None, plotting=False):
        """Registers demonstration scene onto the test scene and uses this registration to transfer the demonstration trajectory

        Args:
            demo: Demonstration that has the demonstration scene and the trajectory to transfer
            test_scene_state: SceneState of the test scene

        Returns:
            The transferred Trajectory
        """
        raise NotImplementedError

class TwoStepRegistrationAndTrajectoryTransferer(RegistrationAndTrajectoryTransferer):
    def transfer(self, demo, test_scene_state, callback=None, plotting=False):
        reg = self.registration_factory.register(demo, test_scene_state, callback=callback)
        test_aug_traj = self.trajectory_transferer.transfer(reg, demo, plotting=plotting)
        return test_aug_traj

class UnifiedRegistrationAndTrajectoryTransferer(RegistrationAndTrajectoryTransferer):
    def __init__(self, registration_factory, trajectory_transferer,
                 alpha=settings.ALPHA,
                 beta_pos=settings.BETA_POS,
                 gamma=settings.GAMMA,
                 use_collision_cost=settings.USE_COLLISION_COST,
                 init_trajectory_transferer=None):
        super(UnifiedRegistrationAndTrajectoryTransferer, self).__init__(registration_factory, trajectory_transferer)
        if not isinstance(registration_factory, registration.TpsRpmRegistrationFactory):
            raise NotImplementedError("UnifiedRegistrationAndTrajectoryTransferer only supports TpsRpmRegistrationFactory")
        if not isinstance(trajectory_transferer, transfer.FingerTrajectoryTransferer):
            raise NotImplementedError("UnifiedRegistrationAndTrajectoryTransferer only supports FingerTrajectoryTransferer")
        self.sim = trajectory_transferer.sim
        self.alpha = alpha
        self.beta_pos = beta_pos
        self.gamma = gamma
        self.use_collision_cost = use_collision_cost
        self.init_trajectory_transferer = init_trajectory_transferer

    def transfer(self, demo, test_scene_state, callback=None, plotting=False):
        reg = self.registration_factory.register(demo, test_scene_state, callback=callback)

        handles = []
        if plotting:
            demo_cloud = demo.scene_state.cloud
            test_cloud = reg.test_scene_state.cloud
            demo_color = demo.scene_state.color
            test_color = reg.test_scene_state.color
            handles.append(self.sim.env.plot3(demo_cloud[:,:3], 2, demo_color if demo_color is not None else (1,0,0)))
            handles.append(self.sim.env.plot3(test_cloud[:,:3], 2, test_color if test_color is not None else (0,0,1)))
            if self.sim.viewer:
              self.sim.viewer.Step()

        active_lr = ""
        for lr in 'lr':
            if lr in demo.aug_traj.lr2arm_traj and sim_util.arm_moved(demo.aug_traj.lr2arm_traj[lr]):
                active_lr += lr
        _, timesteps_rs = sim_util.unif_resample(np.c_[(1./settings.JOINT_LENGTH_PER_STEP) * np.concatenate([demo.aug_traj.lr2arm_traj[lr] for lr in active_lr], axis=1),
                                                       (1./settings.FINGER_CLOSE_RATE) * np.concatenate([demo.aug_traj.lr2finger_traj[lr] for lr in active_lr], axis=1)],
                                                 1.)
        demo_aug_traj_rs = demo.aug_traj.get_resampled_traj(timesteps_rs)

        if self.init_trajectory_transferer:
            warm_init_traj = self.init_trajectory_transferer.transfer(reg, demo, plotting=plotting)

        manip_name = ""
        flr2finger_link_names = []
        flr2demo_finger_pts_trajs_rs = []
        init_traj = np.zeros((len(timesteps_rs),0))
        for lr in active_lr:
            arm_name = {"l":"leftarm", "r":"rightarm"}[lr]
            finger_name = "%s_gripper_l_finger_joint"%lr

            if manip_name:
                manip_name += "+"
            manip_name += arm_name + "+" + finger_name

            if self.init_trajectory_transferer:
                init_traj = np.c_[init_traj, warm_init_traj.lr2arm_traj[lr], warm_init_traj.lr2finger_traj[lr]]
            else:
                init_traj = np.c_[init_traj, demo_aug_traj_rs.lr2arm_traj[lr], demo_aug_traj_rs.lr2finger_traj[lr]]

            if plotting:
                handles.append(self.sim.env.drawlinestrip(demo.aug_traj.lr2ee_traj[lr][:,:3,3], 2, (1,0,0)))
                handles.append(self.sim.env.drawlinestrip(demo_aug_traj_rs.lr2ee_traj[lr][:,:3,3], 2, (1,1,0)))
                transformed_ee_traj_rs = reg.f.transform_hmats(demo_aug_traj_rs.lr2ee_traj[lr])
                handles.append(self.sim.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                if self.sim.viewer:
                  self.sim.viewer.Step()

            flr2demo_finger_pts_traj_rs = sim_util.get_finger_pts_traj(self.sim.robot, lr, (demo_aug_traj_rs.lr2ee_traj[lr], demo_aug_traj_rs.lr2finger_traj[lr]))
            flr2demo_finger_pts_trajs_rs.append(flr2demo_finger_pts_traj_rs)

            flr2transformed_finger_pts_traj_rs = {}
            flr2finger_link_name = {}
            flr2finger_rel_pts = {}
            for finger_lr in 'lr':
                flr2transformed_finger_pts_traj_rs[finger_lr] = reg.f.transform_points(np.concatenate(flr2demo_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                flr2finger_link_name[finger_lr] = "%s_gripper_%s_finger_tip_link"%(lr,finger_lr)
                flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)
            flr2finger_link_names.append(flr2finger_link_name)

            if plotting:
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2demo_finger_pts_traj_rs, (1,1,0)))
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                if self.sim.viewer:
                  self.sim.viewer.Step()

        if not self.init_trajectory_transferer:
            # modify the shoulder joint angle of init_traj to be the limit (highest arm) because this usually gives a better local optima (but this might not be the right thing to do)
            dof_inds = sim_util.dof_inds_from_name(self.sim.robot, manip_name)
            joint_ind = self.sim.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
            init_traj[:,dof_inds.index(joint_ind)] = self.sim.robot.GetDOFLimits([joint_ind])[0][0]

        print "planning joint TPS and finger points trajectory following"
        test_traj, obj_value, tps_rel_pts_costs, tps_cost = planning.decomp_fit_tps_follow_finger_pts_trajs(self.sim.robot, manip_name,
                                                                              flr2finger_link_names, flr2finger_rel_pts,
                                                                              flr2demo_finger_pts_trajs_rs, init_traj,
                                                                              reg.f,
                                                                              use_collision_cost=self.use_collision_cost,
                                                                              start_fixed=False,
                                                                              alpha=self.alpha, beta_pos=self.beta_pos, gamma=self.gamma)

        full_traj = (test_traj, sim_util.dof_inds_from_name(self.sim.robot, manip_name))
        test_aug_traj = demonstration.AugmentedTrajectory.create_from_full_traj(self.sim.robot, full_traj, lr2open_finger_traj=demo_aug_traj_rs.lr2open_finger_traj, lr2close_finger_traj=demo_aug_traj_rs.lr2close_finger_traj)

        if plotting:
            for lr in active_lr:
                flr2new_transformed_finger_pts_traj_rs = {}
                for finger_lr in 'lr':
                    flr2new_transformed_finger_pts_traj_rs[finger_lr] = reg.f.transform_points(np.concatenate(flr2demo_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2new_transformed_finger_pts_traj_rs, (0,1,1)))
                handles.append(self.sim.env.drawlinestrip(test_aug_traj.lr2ee_traj[lr][:,:3,3], 2, (0,0,1)))
                flr2test_finger_pts_traj = sim_util.get_finger_pts_traj(self.sim.robot, lr, full_traj)
                handles.extend(sim_util.draw_finger_pts_traj(self.sim, flr2test_finger_pts_traj, (0,0,1)))
            if self.sim.viewer:
              self.sim.viewer.Step()

        return test_aug_traj
