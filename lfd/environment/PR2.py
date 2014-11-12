from base_robot import BaseRobot

class _BerkeleyPR2(BaseRobot):

    def __init__(self):
        BaseRobot.__init__(self)
        self.including_gripper_finger_collisions = 0
        self.finger_link_names = ["%s_gripper_%s_finger_tip_link" % (lr, flr)
                                  for lr in 'lr' for flr in 'lr']

    def pre_add_objects(self, simulation):
        if not self.including_gripper_finger_collisions:
            self._include_gripper_finger_collisions(simulation)
        self.including_gripper_finger_collisions += 1
    pre_remove_objects = pre_add_objects
    pre_set_state = pre_add_objects

    def post_add_objects(self, simulation):
        self.including_gripper_finger_collisions -= 1
        assert self.including_gripper_finger_collisions >= 0
        if not self.including_gripper_finger_collisions:
            self._exclude_gripper_finger_collisions(simulation)
    post_remove_objects = post_add_objects
    post_set_state = post_add_objects

BerkeleyPR2 = _BerkeleyPR2()