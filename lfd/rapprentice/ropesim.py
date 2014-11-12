import bulletsimpy
import numpy as np
from lfd.rapprentice import math_utils, retiming
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

def retime_traj(robot, inds, traj, max_cart_vel=.02, max_finger_vel=.02, upsample_time=.1):
    """retime a trajectory so that it executes slowly enough for the simulation"""
    cart_traj = np.empty((len(traj), 6))
    finger_traj = np.empty((len(traj),2))
    leftarm, rightarm = robot.GetManipulator("leftarm"), robot.GetManipulator("rightarm")
    with robot:
        for i in range(len(traj)):
            robot.SetDOFValues(traj[i], inds)
            cart_traj[i,:3] = leftarm.GetTransform()[:3,3]
            cart_traj[i,3:] = rightarm.GetTransform()[:3,3]
            finger_traj[i,:1] = robot.GetDOFValues(leftarm.GetGripperIndices())
            finger_traj[i,1:] = robot.GetDOFValues(rightarm.GetGripperIndices())

    times = retiming.retime_with_vel_limits(np.c_[cart_traj, finger_traj], np.r_[np.repeat(max_cart_vel, 6),np.repeat(max_finger_vel,2)])
    times_up = np.linspace(0, times[-1], times[-1]/upsample_time) if times[-1] > upsample_time else times
    traj_up = math_utils.interp2d(times_up, times, traj)
    return traj_up

def observe_cloud(pts, radius, upsample=0, upsample_rad=1):
    """
    If upsample > 0, the number of points along the rope's backbone is resampled to be upsample points
    If upsample_rad > 1, the number of points perpendicular to the backbone points is resampled to be upsample_rad points, around the rope's cross-section
    The total number of points is then: (upsample if upsample > 0 else len(self.rope.GetControlPoints())) * upsample_rad
    """
    if upsample > 0:
        lengths = np.r_[0, np.apply_along_axis(np.linalg.norm, 1, np.diff(pts, axis=0))]
        summed_lengths = np.cumsum(lengths)
        assert len(lengths) == len(pts)
        pts = math_utils.interp2d(np.linspace(0, summed_lengths[-1], upsample), summed_lengths, pts)
    if upsample_rad > 1:
        # add points perpendicular to the points in pts around the rope's cross-section
        vs = np.diff(pts, axis=0) # vectors between the current and next points
        vs /= np.apply_along_axis(np.linalg.norm, 1, vs)[:,None]
        perp_vs = np.c_[-vs[:,1], vs[:,0], np.zeros(vs.shape[0])] # perpendicular vectors between the current and next points in the xy-plane
        perp_vs /= np.apply_along_axis(np.linalg.norm, 1, perp_vs)[:,None]
        vs = np.r_[vs, vs[-1,:][None,:]] # define the vector of the last point to be the same as the second to last one
        perp_vs = np.r_[perp_vs, perp_vs[-1,:][None,:]] # define the perpendicular vector of the last point to be the same as the second to last one
        perp_pts = []
        from openravepy import matrixFromAxisAngle
        for theta in np.linspace(0, 2*np.pi, upsample_rad, endpoint=False): # uniformly around the cross-section circumference
            for (center, rot_axis, perp_v) in zip(pts, vs, perp_vs):
                rot = matrixFromAxisAngle(rot_axis, theta)[:3,:3]
                perp_pts.append(center + rot.T.dot(radius * perp_v))
        pts = np.array(perp_pts)
    return pts

class Simulation(object):
    def __init__(self, env, robot, rope_params):
        self.env = env
        self.robot = robot
        self.bt_env = None
        self.bt_robot = None
        self.rope = None
        self.rope_pts = None
        self.rope_params = rope_params
        self.constraints = {"l": [], "r": []}
        self.constraints_links = {"l": [], "r": []}
    
    def create(self, rope_pts):
        self.bt_env = bulletsimpy.BulletEnvironment(self.env, [])
        self.bt_env.SetGravity([0, 0, -9.8])
        self.bt_robot = self.bt_env.GetObjectByName(self.robot.GetName())
        capsule_rope_params = bulletsimpy.CapsuleRopeParams()
        capsule_rope_params.radius       = self.rope_params.radius
        capsule_rope_params.angStiffness = self.rope_params.angStiffness
        capsule_rope_params.angDamping   = self.rope_params.angDamping
        capsule_rope_params.linDamping   = self.rope_params.linDamping
        capsule_rope_params.angLimit     = self.rope_params.angLimit
        capsule_rope_params.linStopErp   = self.rope_params.linStopErp
        capsule_rope_params.mass         = self.rope_params.mass
        self.rope = bulletsimpy.CapsuleRope(self.bt_env, 'rope', rope_pts, capsule_rope_params)
        self.rope_pts = rope_pts

        cc = trajoptpy.GetCollisionChecker(self.env)
        for gripper_link in [link for link in self.robot.GetLinks() if 'gripper' in link.GetName()]:
            for rope_link in self.rope.GetKinBody().GetLinks():
                cc.ExcludeCollisionPair(gripper_link, rope_link)

        # self.rope.UpdateRave()
        # self.env.UpdatePublishedBodies()
        # trajoptpy.GetViewer(self.env).Idle()

        self.settle()

    def __del__(self):
        if self.rope:
            cc = trajoptpy.GetCollisionChecker(self.env)
            for gripper_link in [link for link in self.robot.GetLinks() if 'gripper' in link.GetName()]:
                for rope_link in self.rope.GetKinBody().GetLinks():
                    cc.IncludeCollisionPair(gripper_link, rope_link)
            # remove all capsule-capsule exclude to prevent memory leak
            # TODO: only interate through the capsule pairs that actually are excluded
            for rope_link0 in self.rope.GetKinBody().GetLinks():
                for rope_link1 in self.rope.GetKinBody().GetLinks():
                    cc.IncludeCollisionPair(rope_link0, rope_link1)
            self.env.Remove(self.rope.GetKinBody())
            self.rope = None
            self.rope_pts = None

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
#         print "settled in %d iterations" % (i+1)

    def observe_cloud(self, upsample=0, upsample_rad=1):
        """
        If upsample > 0, the number of points along the rope's backbone is resampled to be upsample points
        If upsample_rad > 1, the number of points perpendicular to the backbone points is resampled to be upsample_rad points, around the rope's cross-section
        The total number of points is then: (upsample if upsample > 0 else len(self.rope.GetControlPoints())) * upsample_rad
        """
        return observe_cloud(self.rope.GetControlPoints(), self.rope_params.radius, upsample=upsample, upsample_rad=upsample_rad)
    
    def raycast_cloud(self, T_w_k=None, obj=None, z=1., endpoints=0):
        """
        T_w_k: world transform of the kinect. The robot's wide_stereo_optical_frame is used by default. TODO: use the actual kinect's frame
        obj: the BulletObject to do ray test onto. The rope is used by default
        z: length of the rays. 1 meter by default
        endpoints: also return the indices of the points that came from the rope's endpoints. Assumes obj is a rope
        """
        if T_w_k is None:
            T_w_k = self.robot.GetLink("wide_stereo_optical_frame").GetTransform()
        if obj is None:
            obj = self.rope
        
        # camera's parameters
        cx = 320.-.5
        cy = 240.-.5
        f = 525. # focal length
        w = 640.
        h = 480.
        
        pixel_ij = np.array(np.meshgrid(np.arange(w), np.arange(h))).T.reshape((-1,2)) # all pixel positions
        rayTos = z * np.c_[(pixel_ij - np.array([cx, cy])) / f, np.ones(pixel_ij.shape[0])]
        rayFroms = np.zeros_like(rayTos)
        # transform the rays from the camera frame to the world frame
        rayTos = rayTos.dot(T_w_k[:3,:3].T) + T_w_k[:3,3]
        rayFroms = rayFroms.dot(T_w_k[:3,:3].T) + T_w_k[:3,3]

        ray_collisions = self.bt_env.RayTest(rayFroms, rayTos, obj)

        pts = np.empty((len(ray_collisions), 3))
        for i, ray_collision in enumerate(ray_collisions):
            pts[i,:] = ray_collision.pt

        if endpoints:
            links = self.rope.GetKinBody().GetLinks()
            end_links = links[:endpoints] + links[-endpoints:]
            endpoint_inds = np.zeros(len(pts), dtype=bool)
            for i, ray_collision in enumerate(ray_collisions):
                if ray_collision.link in end_links:
                    endpoint_inds[i] = True
            return pts, endpoint_inds
        
        return pts

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
            i_cnt = i_node
            for geom in rope_links[i_cnt].GetGeometries():
                geom.SetDiffuseColor([1.,0.,0.])
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
            self.constraints_links[lr].append(rope_links[i_cnt])

        return True

    def release_rope(self, lr):
#         print 'RELEASE: %s (%d constraints)' % (lr, len(self.constraints[lr]))
        for c in self.constraints[lr]:
            self.bt_env.RemoveConstraint(c)
        rope_links = self.rope.GetKinBody().GetLinks()
        for link in self.constraints_links[lr]:
            for geom in link.GetGeometries():
                geom.SetDiffuseColor([1.,1.,1.])
        self.constraints[lr] = []
        self.constraints_links[lr] = []
    
    def is_grabbing_rope(self, lr):
        return bool(self.constraints[lr])
