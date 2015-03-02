from __future__ import division

from lfd.rapprentice.yes_or_no import yes_or_no
from lfd.demonstration import demonstration
try:
    from lfd.rapprentice import pr2_trajectories
except:
    print "Couldn't import ros stuff"

class RobotWorld(object):
    def __init__(self):
        raise NotImplementedError
    
    def observe_cloud(self):
        raise NotImplementedError
    
    def open_gripper(self):
        raise NotImplementedError
    
    def close_gripper(self):
        raise NotImplementedError
    
    def execute_trajectory(self):
        raise NotImplementedError

class RealRobotWorld(RobotWorld):
    def __init__(self, pr2):
        self.pr2 = pr2
        self.robot = pr2.robot
        self.env = pr2.env
    
    def observe_cloud(self):
        raise NotImplementedError
    
    def open_gripper(self, lr, target_val=0.54800022, step_viewer=0):
        gripper = {"l":self.pr2.lgrip, "r":self.pr2.rgrip}[lr]
        gripper.set_angle(target_val)
        self.pr2.join_all()
    
    def close_gripper(self, lr, step_viewer=0):
        gripper = {"l":self.pr2.lgrip, "r":self.pr2.rgrip}[lr]
        gripper.set_angle(0.0)
    
        self.pr2.join_all()
    def execute_trajectory(self, full_traj, prompt=False, step_viewer=0, interactive=False, sim_callback=None):
        if not prompt or yes_or_no("execute?"):
            # TODO consider finger trajectory
            aug_traj = demonstration.AugmentedTrajectory.create_from_full_traj(self.pr2.robot, full_traj)
            bodypart2traj = {}
            for lr, arm_traj in aug_traj.lr2arm_traj.items():
                part_name = {"l":"larm", "r":"rarm"}[lr]
                bodypart2traj[part_name] = arm_traj
            pr2_trajectories.follow_body_traj(self.pr2, bodypart2traj, speed_factor=0.5)
