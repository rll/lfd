import numpy as np

class RosToRave(object):
    
    "take in ROS joint_states messages and use it to update an openrave robot"
    
    def __init__(self, robot, ros_names):
        self.initialized = False
        
        self.ros_names  = ros_names
        inds_ros2rave = np.array([robot.GetJointIndex(name) for name in self.ros_names])
        self.good_ros_inds = np.flatnonzero(inds_ros2rave != -1) # ros joints inds with matching rave joint
        self.rave_inds = inds_ros2rave[self.good_ros_inds] # openrave indices corresponding to those joints

    def convert(self, ros_values):
        return [ros_values[i_ros] for i_ros in self.good_ros_inds]
    def set_values(self, robot, ros_values):
        rave_values = [ros_values[i_ros] for i_ros in self.good_ros_inds]        
        robot.SetDOFValues(rave_values,self.rave_inds, 0)
        