import bulletsimpy
import numpy as np
from rapprentice import math_utils, retiming
import trajoptpy

def transform(hmat, p):
    return hmat[:3,:3].dot(p) + hmat[:3,3]

def in_grasp_region(robot, lr, pt):
    tol = .00

    manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
    manip = robot.GetManipulator(manip_name)
    l_finger = robot.GetLink("%s_gripper_l_finger_tip_link"%lr)
    r_finger = robot.GetLink("%s_gripper_r_finger_tip_link"%lr)

    def on_inner_side(pt, finger_lr):
        finger = l_finger
        closing_dir = np.cross(manip.GetLocalToolDirection(), [-1, 0, 0])
        local_inner_pt = np.array([0.234402, -0.299, 0])/20.
        if finger_lr == "r":
            finger = r_finger
            closing_dir *= -1
            local_inner_pt[1] *= -1
        inner_pt = transform(finger.GetTransform(), local_inner_pt)
        return manip.GetTransform()[:3,:3].dot(closing_dir).dot(pt - inner_pt) > 0

    # check that pt is behind the gripper tip
    pt_local = transform(np.linalg.inv(manip.GetTransform()), pt)
    if pt_local[2] > .03 + tol:
        return False

    # check that pt is within the finger width
    if abs(pt_local[0]) > .01 + tol:
        return False

    # check that pt is between the fingers
    if not on_inner_side(pt, "l") or not on_inner_side(pt, "r"):
        return False

    return True

def retime_traj(robot, inds, traj, max_cart_vel=.02, upsample_time=.1):
    """retime a trajectory so that it executes slowly enough for the simulation"""
    cart_traj = np.empty((len(traj), 6))
    leftarm, rightarm = robot.GetManipulator("leftarm"), robot.GetManipulator("rightarm")
    with robot:
        for i in range(len(traj)):
            robot.SetDOFValues(traj[i], inds)
            cart_traj[i,:3] = leftarm.GetTransform()[:3,3]
            cart_traj[i,3:] = rightarm.GetTransform()[:3,3]

    times = retiming.retime_with_vel_limits(cart_traj, np.repeat(max_cart_vel, 6))
    times_up = np.linspace(0, times[-1], times[-1]/upsample_time) if times[-1] > upsample_time else times
    traj_up = math_utils.interp2d(times_up, times, traj)
    return traj_up


class Simulation(object):
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot
        self.bt_env = None
        self.bt_robot = None
        self.rope = None
        self.constraints = {"l": [], "r": []}

        self.rope_params = bulletsimpy.CapsuleRopeParams()
        self.rope_params.radius = 0.005
        self.rope_params.angStiffness = .1
        self.rope_params.angDamping = 1
        self.rope_params.linDamping = .75
        self.rope_params.angLimit = .4
        self.rope_params.linStopErp = .2

    def create(self, rope_pts):
        self.bt_env = bulletsimpy.BulletEnvironment(self.env, [])
        self.bt_env.SetGravity([0, 0, -9.8])
        self.bt_robot = self.bt_env.GetObjectByName(self.robot.GetName())
        self.rope = bulletsimpy.CapsuleRope(self.bt_env, 'rope', rope_pts, self.rope_params)

        # self.rope.UpdateRave()
        # self.env.UpdatePublishedBodies()
        # trajoptpy.GetViewer(self.env).Idle()

        self.settle()

    def step(self):
        self.bt_robot.UpdateBullet()
        self.bt_env.Step(.01, 200, .005)
        self.rope.UpdateRave()
        self.env.UpdatePublishedBodies()

    def settle(self, max_steps=100, tol=.001, animate=False):
        """Keep stepping until the rope doesn't move, up to some tolerance"""
        prev_nodes = self.rope.GetNodes()
        for i in range(max_steps):
            self.bt_env.Step(.01, 200, .005)
            if animate:
                self.rope.UpdateRave()
                self.env.UpdatePublishedBodies()
            if i % 10 == 0 and i != 0:
                curr_nodes = self.rope.GetNodes()
                diff = np.sqrt(((curr_nodes - prev_nodes)**2).sum(axis=1))
                if diff.max() < tol:
                    break
                prev_nodes = curr_nodes
        self.rope.UpdateRave()
        self.env.UpdatePublishedBodies()
        print "settled in %d iterations" % (i+1)

    def observe_cloud(self, upsample=0):
        pts = self.rope.GetControlPoints()
        if upsample == 0:
            return pts
        lengths = np.r_[0, self.rope.GetHalfHeights() * 2]
        summed_lengths = np.cumsum(lengths)
        assert len(lengths) == len(pts)
        return math_utils.interp2d(np.linspace(0, summed_lengths[-1], upsample), summed_lengths, pts)

    def grab_rope(self, lr):
        nodes, ctl_pts = self.rope.GetNodes(), self.rope.GetControlPoints()

        graspable_nodes = np.array([in_grasp_region(self.robot, lr, n) for n in nodes])
        graspable_ctl_pts = np.array([in_grasp_region(self.robot, lr, n) for n in ctl_pts])
        graspable_inds = np.flatnonzero(np.logical_or(graspable_nodes, np.logical_or(graspable_ctl_pts[:-1], graspable_ctl_pts[1:])))
        print 'graspable inds for %s: %s' % (lr, str(graspable_inds))
        if len(graspable_inds) == 0:
            return False

        robot_link = self.robot.GetLink("%s_gripper_l_finger_tip_link"%lr)
        rope_links = self.rope.GetKinBody().GetLinks()
        for i_node in graspable_inds:
            for i_cnt in range(max(0, i_node-1), min(len(nodes), i_node+2)):
                cnt = self.bt_env.AddConstraint({
                    "type": "generic6dof",
                    "params": {
                        "link_a": robot_link,
                        "link_b": rope_links[i_cnt],
                        "frame_in_a": np.linalg.inv(robot_link.GetTransform()).dot(rope_links[i_cnt].GetTransform()),
                        "frame_in_b": np.eye(4),
                        "use_linear_reference_frame_a": False,
                        "stop_erp": .8,
                        "stop_cfm": .1,
                        "disable_collision_between_linked_bodies": True,
                    }
                })
                self.constraints[lr].append(cnt)

        return True

    def release_rope(self, lr):
        print 'RELEASE: %s (%d constraints)' % (lr, len(self.constraints[lr]))
        for c in self.constraints[lr]:
            self.bt_env.RemoveConstraint(c)
        self.constraints[lr] = []
