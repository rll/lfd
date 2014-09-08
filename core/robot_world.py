from __future__ import division

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
