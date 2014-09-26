from __future__ import division

import openravepy
import trajoptpy, bulletsimpy
from rapprentice import animate_traj, ropesim, eval_util
import numpy as np
import demonstration, simulation_object, sim_util

class LfdEnvironment(object):
    def __init__(self, world, sim, downsample_size=0):
        """Inits LfdEnvironment
        
        Args:
            world: RobotWorld
            sim: StaticSimulation that contains a robot
            downsample_size: if downsample_size is positive, the clouds are downsampled to a voxel size of downsample_size, else they are not downsampled
        """
        self.world = world
        self.sim = sim
        self.downsample_size = downsample_size
    
    def execute_augmented_trajectory(self, aug_traj, step_viewer=1, interactive=False, sim_callback=None, check_feasible=False):
        open_or_close_finger_traj = np.zeros(aug_traj.n_steps, dtype=bool)
        if aug_traj.lr2open_finger_traj is not None:
            for lr in aug_traj.lr2open_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2open_finger_traj[lr])
        if aug_traj.lr2close_finger_traj is not None:
            for lr in aug_traj.lr2close_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2close_finger_traj[lr])
        open_or_close_inds = np.where(open_or_close_finger_traj)[0]
        
        traj, dof_inds = aug_traj.get_full_traj(self.sim.robot)
        feasible = True
        misgrasp = False
        lr2gripper_open = {'l':True, 'r':True}
        for (start_ind, end_ind) in zip(np.r_[0, open_or_close_inds], np.r_[open_or_close_inds+1, aug_traj.n_steps]):
            if aug_traj.lr2open_finger_traj is not None:
                for lr in aug_traj.lr2open_finger_traj.keys():
                    if aug_traj.lr2open_finger_traj[lr][start_ind]:
                        target_val = None
                        joint_ind = self.sim.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                        if joint_ind in dof_inds:
                            target_val = traj[start_ind, dof_inds.index(joint_ind)]
                        self.world.open_gripper(lr, target_val=target_val, step_viewer=step_viewer)
                        lr2gripper_open[lr] = True
            if aug_traj.lr2close_finger_traj is not None:
                for lr in aug_traj.lr2close_finger_traj.keys():
                    if aug_traj.lr2close_finger_traj[lr][start_ind]:
                        n_cnts = len(self.sim.constraints[lr])
                        self.world.close_gripper(lr, step_viewer=step_viewer)
                        misgrasp |= len(self.sim.constraints[lr]) == n_cnts
                        lr2gripper_open[lr] = False
            # don't execute trajectory for finger joint if the corresponding gripper is closed
            active_inds = np.ones(len(dof_inds), dtype=bool)
            for lr in 'lr':
                if not lr2gripper_open[lr]:
                    joint_ind = self.sim.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                    if joint_ind in dof_inds:
                        active_inds[dof_inds.index(joint_ind)] = False
            miniseg_traj = traj[start_ind:end_ind, active_inds]
            miniseg_dof_inds = list(np.asarray(dof_inds)[active_inds])
            full_traj = (miniseg_traj, miniseg_dof_inds)
            feasible &= eval_util.traj_is_safe(self.sim, full_traj, 0)
            if check_feasible and not feasible:
                break
            self.world.execute_trajectory(full_traj, step_viewer=step_viewer, interactive=interactive, sim_callback=sim_callback)
        return feasible, misgrasp
    
    def observe_scene(self):
        full_cloud = self.world.observe_cloud()
        return demonstration.SceneState(full_cloud, downsample_size=self.downsample_size)

class GroundTruthRopeLfdEnvironment(LfdEnvironment):
    def __init__(self, world, sim, upsample=0, upsample_rad=1, downsample_size=0):
        """Inits GroundTruthRopeLfdEnvironment
        
        Args:
            world: RobotWorld
            sim: DynamicSimulation that should containing exactly one rope when observe_scene is called
            downsample_size: if downsample_size is positive, the clouds are downsampled to a voxel size of downsample_size, else they are not downsampled
        """
        super(GroundTruthRopeLfdEnvironment, self).__init__(world, sim, downsample_size=downsample_size)
        
        self.upsample = upsample
        self.upsample_rad = upsample_rad
    
    def observe_scene(self):
        rope_sim_objs = [sim_obj for sim_obj in self.sim.sim_objs if isinstance(sim_obj, simulation_object.RopeSimulationObject)]
        assert len(rope_sim_objs) == 1
        rope_sim_obj = rope_sim_objs[0]
        return demonstration.GroundTruthRopeSceneState(rope_sim_obj.rope.GetControlPoints(), 
                                                       rope_sim_obj.rope_params.radius, 
                                                       upsample=self.upsample, 
                                                       upsample_rad=self.upsample_rad, 
                                                       downsample_size=self.downsample_size)

class GroundTruthBoxLfdEnvironment(LfdEnvironment):
    def __init__(self, world, sim, box0pos, box1pos, box1pos_2, box_width, box_top_height, bottom_height, downsample_size=0):
        super(GroundTruthBoxLfdEnvironment, self).__init__(world, sim,downsample_size=downsample_size)
        self.box0pos = box0pos
        self.box1pos = box1pos
        self.box1pos_2 = box1pos_2
        self.box_width = box_width
        self.box_top_height = box_top_height
        self.bottom_height = bottom_height

    def observe_scene(self, scene, ground_truth=False):
        if ground_truth:
            pts = np.zeros((0,3))
            box0_x,box0_y = self.box0pos[0],self.box0pos[1]
            box1_x,box1_y = self.box1pos[0],self.box1pos[1]
            box1_x_2,box1_y_2 = self.box1pos_2[0],self.box1pos_2[1]
            box_width = self.box_width
            box_top_height = self.box_top_height

            pts = np.r_[pts,np.array([[box0_x-box_width,box0_y-box_width,box_top_height],
            [box0_x-box_width,box0_y+box_width,box_top_height],
            [box0_x+box_width,box0_y-box_width,box_top_height],
            [box0_x+box_width,box0_y+box_width,box_top_height]])]

            if scene == ("demonstration"):
                pts = np.r_[pts,np.array([[box1_x-box_width,box1_y-box_width,box_top_height],
                [box1_x-box_width,box1_y+box_width,box_top_height],
                [box1_x+box_width,box1_y-box_width,box_top_height],
                [box1_x+box_width,box1_y+box_width,box_top_height]])]
            else:
                pts = np.r_[pts,np.array([[box1_x_2-box_width,box1_y_2-box_width,box_top_height],
                [box1_x_2-box_width,box1_y_2+box_width,box_top_height],
                [box1_x_2+box_width,box1_y_2-box_width,box_top_height],
                [box1_x_2+box_width,box1_y_2+box_width,box_top_height]])]

            box_top_height = self.bottom_height

            pts = np.r_[pts,np.array([[box0_x-box_width,box0_y-box_width,box_top_height],
            [box0_x-box_width,box0_y+box_width,box_top_height],
            [box0_x+box_width,box0_y-box_width,box_top_height],
            [box0_x+box_width,box0_y+box_width,box_top_height]])]

            if scene == ("demonstration"):
                pts = np.r_[pts,np.array([[box1_x-box_width,box1_y-box_width,box_top_height],
                [box1_x-box_width,box1_y+box_width,box_top_height],
                [box1_x+box_width,box1_y-box_width,box_top_height],
                [box1_x+box_width,box1_y+box_width,box_top_height]])]
            else:
                pts = np.r_[pts,np.array([[box1_x_2-box_width,box1_y_2-box_width,box_top_height],
                [box1_x_2-box_width,box1_y_2+box_width,box_top_height],
                [box1_x_2+box_width,box1_y_2-box_width,box_top_height],
                [box1_x_2+box_width,box1_y_2+box_width,box_top_height]])]


            return demonstration.SceneState(pts,downsample_size=0)
        else:
            full_cloud = self.world.observe_cloud()
            return demonstration.SceneState(full_cloud, downsample_size=self.downsample_size)

class RecordingLfdEnvironment(GroundTruthRopeLfdEnvironment):
    def __init__(self, world, sim, upsample=0, upsample_rad=1, downsample_size=0):
        super(RecordingLfdEnvironment, self).__init__(world, sim, upsample=upsample, upsample_rad=upsample_rad, downsample_size=downsample_size)
        self.cur_step_states = []
    
    def execute_augmented_trajectory(self, aug_traj, step_viewer=1, interactive=False, sim_callback=None):
        self.cur_step_states = []
        if sim_callback is None:
            def sim_cb(i):
                rope_sim_objs = [sim_obj for sim_obj in self.sim.sim_objs if isinstance(sim_obj, simulation_object.RopeSimulationObject)]
                assert len(rope_sim_objs) == 1
                rope_sim_obj = rope_sim_objs[0]
                cur_state = demonstration.TimestepState(rope_sim_obj.rope.GetControlPoints(), self.sim.robot, step=i)
                self.cur_step_states.append(cur_state)
                self.sim.step()
            sim_callback = sim_cb
        return super(RecordingLfdEnvironment, self).execute_augmented_trajectory(aug_traj, 
                                                                                 step_viewer=step_viewer, 
                                                                                 interactive=interactive, 
                                                                                 sim_callback=sim_callback)
    
    def observe_scene(self):
        rope_sim_objs = [sim_obj for sim_obj in self.sim.sim_objs if isinstance(sim_obj, simulation_object.RopeSimulationObject)]
        assert len(rope_sim_objs) == 1
        rope_sim_obj = rope_sim_objs[0]
        return demonstration.RecordingRopePositionsSceneState(rope_sim_obj.rope.GetControlPoints(), 
                                                              self.cur_step_states, 
                                                              rope_sim_obj.rope_params.radius, 
                                                              upsample=self.upsample, 
                                                              upsample_rad=self.upsample_rad, 
                                                              downsample_size=self.downsample_size)