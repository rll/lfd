import trajoptpy

class BaseRobot():
    def __init__(self):
        self.finger_link_names = None

    def _exclude_gripper_finger_collisions(self, simulation):
        if not simulation.robot:
            return
        cc = trajoptpy.GetCollisionChecker(simulation.env)
        for finger_link_name in self.finger_link_names:
            finger_link = simulation.robot.GetLink(finger_link_name)
            for sim_obj in simulation.sim_objs:
                for bt_obj in sim_obj.get_bullet_objects():
                    for link in bt_obj.GetKinBody().GetLinks():
                        cc.ExcludeCollisionPair(finger_link, link)

    def _include_gripper_finger_collisions(self, simulation):
        if not simulation.robot:
            return
        cc = trajoptpy.GetCollisionChecker(simulation.env)
        for finger_link_name in self.finger_link_names:
            finger_link = simulation.robot.GetLink(finger_link_name)
            for sim_obj in simulation.sim_objs:
                for bt_obj in sim_obj.get_bullet_objects():
                    for link in bt_obj.GetKinBody().GetLinks():
                        cc.IncludeCollisionPair(finger_link, link)