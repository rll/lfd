from __future__ import division

import numpy as np
import trajoptpy
from rapprentice import ropesim, resampling
from rapprentice import math_utils as mu
import sim_util # TODO fold in sim_util function into LfdEnvironment
from tpsopt.constants import MAX_CLD_SIZE

import IPython as ipy

class Demonstration(object):
    def __init__(self, name, scene_state, aug_traj):
        """Inits Demonstration
        
        Args:
            name: demonstration name, which is the same as the ones used for indexing the demonstrations
            scene_state: demonstration SceneState
            aug_traj: demonstration AugmentedTrajectory. Only lr2ee_traj and lr2finger_traj are required for the demonstration AugmentedTrajectory. If lr2arm_traj is specified, it might be used for initializing the trajectory optimization.
        """
        self.name = name
        self.scene_state = scene_state
        assert aug_traj.lr2ee_traj is not None
        assert aug_traj.lr2finger_traj is not None
        self.aug_traj = aug_traj
    
    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__.__name__, self.name, self.scene_state.__repr__(), self.aug_traj.__repr__())


class SceneState(object):
    ids = set()
    def __init__(self, full_cloud, id=None, full_color=None, downsample_size=0):
        """Inits SceneState
        
        Args:
            full_cloud: full (i.e. not downsampled) cloud
            id: unique id for this SceneState
            full_color: colors for the respective points of full_cloud. Should have the same dimensions as full_cloud
            downsample_size: if downsample_size is positive, the full cloud and color are downsampled to a voxel size of downsample_size, else they are not downsampled
        """
        self.full_cloud = full_cloud
        self.full_color = full_color
        if downsample_size > 0:
            global clouds
            from rapprentice import clouds
            if full_color is not None:
                cloud_color = clouds.downsample(np.c_[full_cloud, full_color], downsample_size)
                self.cloud = cloud_color[:,:3]
                self.color = cloud_color[:,3:]
            else:
                self.cloud = clouds.downsample(full_cloud, downsample_size)
                while self.cloud.shape[0] > MAX_CLD_SIZE:
                    downsample_size += .001
                    self.cloud = clouds.downsample(full_cloud, downsample_size)
                self.color = None
        else:
            self.cloud = full_cloud
            self.color = full_color
        if id is None:
            self.id = SceneState.get_unique_id()
        else:
            self.id = id

    @staticmethod
    def get_unique_id():
        id = len(SceneState.ids)
        assert id not in SceneState.ids
        SceneState.ids.add(id)
        return id
    
    def __repr__(self):
        return "%s(..., id=%i)" % (self.__class__.__name__, self.id)

class GroundTruthRopeSceneState(SceneState):
    def __init__(self, rope_nodes, radius, upsample=0, upsample_rad=1, downsample_size=0):
        full_cloud = ropesim.observe_cloud(rope_nodes, radius, upsample=upsample, upsample_rad=upsample_rad)
        super(GroundTruthRopeSceneState, self).__init__(full_cloud, downsample_size=downsample_size)
        self.rope_nodes = rope_nodes
        self.crossing_info = None #TODO: optionally compute/load cached crossing_info

class RecordingRopePositionsSceneState(SceneState):
    def __init__(self, rope_nodes, history, radius, upsample=0, upsample_rad=1, downsample_size=0):
        full_cloud = ropesim.observe_cloud(rope_nodes, radius, upsample=upsample, upsample_rad=upsample_rad)
        super(RecordingRopePositionsSceneState, self).__init__(full_cloud, downsample_size=downsample_size)
        self.rope_nodes = rope_nodes
        self.crossing_info = None #TODO: optionally compute/load cached crossing_info
        self.history = history #TimestepStates at every timestep of most recent trajectory execution


class TimestepState(object):
    def __init__(self, rope_nodes, robot, step=None):
        self.rope_nodes = rope_nodes
        self.step = step
        self.time = 0
        self.manip_trajs = {}
        self.gripper_vals = {}
        for manip in robot.GetManipulators():
            self.manip_trajs[manip.GetName()] = (manip.GetArmDOFValues(), manip.GetArmIndices())
        for lr in 'lr':
            joint_ind = robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
            val = robot.GetDOFValues([joint_ind])[0]
            self.gripper_vals["%s_gripper_l_finger_joint"%lr] = val #this int argument apparently does nothing (?)
            print val

class AugmentedTrajectory(object):
    def __init__(self, lr2arm_traj=None, lr2finger_traj=None, lr2ee_traj=None, lr2open_finger_traj=None, lr2close_finger_traj=None):
        """Inits AugmentedTrajectory
        
        Args:
            lr2arm_traj: dict that maps from 'l' and/or 'r' to the left arm's and/or right arm's joint angle trajectory
            lr2finger_traj: dict that maps to the left gripper's and/or right gripper's finger joint angle trajectory
            lr2ee_traj: dict that maps to the left gripper's and/or right gripper's end-effector trajectory (i.e. a numpy.array of homogeneous matrices)
            lr2open_finger_traj: dict that maps to a boolean vector indicating whether there is a opening action for the left gripper and/or right gripper for every time step. By default, there is no action for all time steps.
            lr2close_finger_traj: same as lr2open_finger_traj but for closing action
        """
        # make sure all trajs have the same number of steps
        self.n_steps = None
        for lr2traj in [lr2arm_traj, lr2finger_traj, lr2ee_traj, lr2open_finger_traj, lr2close_finger_traj]:
            if lr2traj is None:
                continue
            for lr in lr2traj.keys():
                if self.n_steps is None:
                    self.n_steps = lr2traj[lr].shape[0]
                else:
                    assert lr2traj[lr].shape[0] == self.n_steps

        self.lr2arm_traj = lr2arm_traj
        self.lr2finger_traj = lr2finger_traj
        
        self.lr2ee_traj = lr2ee_traj
        
        if lr2open_finger_traj is None:
            self.lr2open_finger_traj = np.zeros(self.n_steps, dtype=bool)
        else:
            self.lr2open_finger_traj = lr2open_finger_traj
        if lr2close_finger_traj is None:
            self.lr2close_finger_traj = np.zeros(self.n_steps, dtype=bool)
        else:
            self.lr2close_finger_traj = lr2close_finger_traj
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            for (lr2traj, other_lr2traj) in [(self.lr2arm_traj, other.lr2arm_traj), (self.lr2finger_traj, other.lr2finger_traj),
                                             (self.lr2ee_traj, other.lr2ee_traj),
                                             (self.lr2open_finger_traj, other.lr2open_finger_traj), (self.lr2close_finger_traj, other.lr2close_finger_traj)]:
                if lr2traj is None:
                    if other_lr2traj is None:
                        continue
                    else:
                        return False
                if set(lr2traj.keys()) != set(other_lr2traj.keys()):
                    return False
                for lr in lr2traj.keys():
                    if np.any(lr2traj[lr] != other_lr2traj[lr]):
                        return False
            return True
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    @staticmethod
    def create_from_full_traj(robot, full_traj, lr2open_finger_traj=None, lr2close_finger_traj=None):
        traj, dof_inds = full_traj
        lr2arm_traj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            arm_inds = robot.GetManipulator(manip_name).GetArmIndices()
            if set(arm_inds).intersection(set(dof_inds)):
                if not set(arm_inds).issubset(set(dof_inds)):
                    raise RuntimeError("Cannot create AugmentedTrajectory from incomplete full_traj")
                arm_traj = np.zeros((traj.shape[0], len(arm_inds)))
                for (j,arm_ind) in enumerate(arm_inds):
                    arm_traj[:,j] = traj[:,dof_inds.index(arm_ind)]
                lr2arm_traj[lr] = arm_traj
        lr2finger_traj = {}
        for lr in 'lr':
            finger_ind = robot.GetJointIndex("%s_gripper_l_finger_joint"%lr)
            if finger_ind in dof_inds:
                lr2finger_traj[lr] = traj[:,dof_inds.index(finger_ind)][:,None]
        lr2ee_traj = {}    
        for lr in lr2arm_traj.keys():
            lr2ee_traj[lr] = sim_util.get_ee_traj(robot, lr, lr2arm_traj[lr])
        return AugmentedTrajectory(lr2arm_traj=lr2arm_traj, lr2finger_traj=lr2finger_traj, lr2ee_traj=lr2ee_traj, lr2open_finger_traj=lr2open_finger_traj, lr2close_finger_traj=lr2close_finger_traj)
    
    def get_full_traj(self, robot):
        """
        TODO: remove sim_util.get_full_traj
        """
        trajs = []
        dof_inds = []
        for lr in 'lr':
            if lr in self.lr2arm_traj:
                arm_traj = self.lr2arm_traj[lr]
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                trajs.append(arm_traj)
                dof_inds.extend(robot.GetManipulator(manip_name).GetArmIndices())
        for lr in 'lr':
            if lr in self.lr2finger_traj:
                finger_traj = self.lr2finger_traj[lr]
                trajs.append(finger_traj)
                dof_inds.append(robot.GetJointIndex("%s_gripper_l_finger_joint"%lr))
        if len(trajs) > 0:
            full_traj = (np.concatenate(trajs, axis=1), dof_inds)
        else:
            full_traj = (np.zeros((0,0)), [])
        return full_traj
    
    def get_resampled_traj(self, timesteps_rs):
        lr2arm_traj_rs = None if self.lr2arm_traj is None else {}
        lr2finger_traj_rs = None if self.lr2finger_traj is None else {}
        for (lr2traj_rs, self_lr2traj) in [(lr2arm_traj_rs, self.lr2arm_traj), (lr2finger_traj_rs, self.lr2finger_traj)]:
            if self_lr2traj is None:
                continue
            for lr in self_lr2traj.keys():
                lr2traj_rs[lr] = mu.interp2d(timesteps_rs, np.arange(len(self_lr2traj[lr])), self_lr2traj[lr])

        if self.lr2ee_traj is None:
            lr2ee_traj_rs = None
        else:
            lr2ee_traj_rs = {}
            for lr in self.lr2ee_traj.keys():
                lr2ee_traj_rs[lr] = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(self.lr2ee_traj[lr])), self.lr2ee_traj[lr]))
        
        lr2open_finger_traj_rs = None if self.lr2open_finger_traj is None else {}
        lr2close_finger_traj_rs = None if self.lr2close_finger_traj is None else {}
        for (lr2oc_finger_traj_rs, self_lr2oc_finger_traj) in [(lr2open_finger_traj_rs, self.lr2open_finger_traj), (lr2close_finger_traj_rs, self.lr2close_finger_traj)]:
            if self_lr2oc_finger_traj is None:
                continue
            for lr in self_lr2oc_finger_traj.keys():
                self_oc_finger_traj = self_lr2oc_finger_traj[lr]
                self_oc_inds = np.where(self_oc_finger_traj)[0]
                oc_inds_rs = np.abs(timesteps_rs[:,None] - self_oc_inds[None,:]).argmin(axis=0)
                oc_finger_traj_rs = np.zeros(len(timesteps_rs), dtype=bool)
                oc_finger_traj_rs[oc_inds_rs] = True
                lr2oc_finger_traj_rs[lr] = oc_finger_traj_rs
        
        return AugmentedTrajectory(lr2arm_traj=lr2arm_traj_rs, lr2finger_traj=lr2finger_traj_rs, lr2ee_traj=lr2ee_traj_rs, lr2open_finger_traj=lr2open_finger_traj_rs, lr2close_finger_traj=lr2close_finger_traj_rs)
    
    def __repr__(self):
        return "%s(...)" % (self.__class__.__name__)
