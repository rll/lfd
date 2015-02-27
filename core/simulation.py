from __future__ import division

import openravepy
import trajoptpy, bulletsimpy
from rapprentice import animate_traj, ropesim
import numpy as np
from robot_world import RobotWorld
import sim_util
import importlib

from core.simulation_object import XmlSimulationObject, BoxSimulationObject, RopeSimulationObject, \
    CylinderSimulationObject, CoilSimulationObject, ContainerSimulationObject


import ipdb

class StaticSimulation(object):
    def __init__(self, env=None):
        if env is not None:
            self.env = env
        else:
            self.env = openravepy.Environment()
            self.env.StopSimulation()
        self.sim_objs = []
        self.robot = None
        self.__viewer = None # don't use this. use viewer instead, which is automatically set the viewer if the it already exists

    def add_objects(self, sim_objs, consider_finger_collisions=True):
        if consider_finger_collisions:
            self._include_gripper_finger_collisions()
        n_robots = len(self.env.GetRobots())
        for sim_obj in sim_objs:
            if sim_obj.dynamic:
                raise RuntimeError("Dynamic object can't be added to StaticSimulation")
            else:
                self.sim_objs.append(sim_obj)
                sim_obj.add_to_env(self)
        if len(self.env.GetRobots()) > n_robots:
            if self.robot:
                raise NotImplementedError("Behavior for adding more than one robot has not been defined")
            self.robot = self.env.GetRobots()[-1]
        if consider_finger_collisions:
            self._exclude_gripper_finger_collisions()

    def remove_objects(self, sim_objs, consider_finger_collisions=True):
        if consider_finger_collisions:
            self._include_gripper_finger_collisions()
        for sim_obj in sim_objs:
            if sim_obj.dynamic:
                raise RuntimeError("Dynamic object can't be removed from StaticSimulation")
            else:
                self.sim_objs.remove(sim_obj)
                sim_obj.remove_from_env()
        if self.robot and self.robot not in self.env.GetRobots():
            raise NotImplementedError("Behavior for removing robots has not been defined")
        if consider_finger_collisions:
            self._exclude_gripper_finger_collisions()

    def get_state(self):
        constr_infos = [sim_obj._get_constructor_info() for sim_obj in self.sim_objs]

        states = {}
        for sim_obj in self.sim_objs:
            state_key = "".join(sim_obj.names)
            assert state_key not in states
            states[state_key] = sim_obj.get_state()
        states["dof_limits"] = np.asarray(self.robot.GetDOFLimits())
        states["dof_values"] = self.robot.GetDOFValues()

        sim_state = (constr_infos, states)
        return sim_state

    def set_state(self, sim_state):
        constr_infos, states = sim_state

        cur_constr_infos = [sim_obj._get_constructor_info() for sim_obj in self.sim_objs]

        constr_infos_to_remove = [constr_info for constr_info in cur_constr_infos if constr_info not in constr_infos]
        constr_infos_to_add = [constr_info for constr_info in constr_infos if constr_info not in cur_constr_infos]
        sim_objs_to_add = []
        sim_objs_to_remove = []
        for constr_info in constr_infos_to_remove:
            sim_obj = self.sim_objs[cur_constr_infos.index(constr_info)]
            sim_objs_to_remove.append(sim_obj)
        for constr_info in constr_infos_to_add:
            ((class_name, class_module), args, kwargs) = constr_info
            class_module = importlib.import_module(class_module)
            c = getattr(class_module, class_name)
            sim_objs_to_add.append(c(*args, **kwargs))
        self.remove_objects(sim_objs_to_remove)
        self.add_objects(sim_objs_to_add)

        # the states should have one and only one state for every sim_obj and dof info
        states_keys = ["".join(sim_obj.names) for sim_obj in self.sim_objs] + ["dof_limits", "dof_values"]
        assert set(states_keys) == set(states.keys())
        for sim_obj in self.sim_objs:
            state_key = "".join(sim_obj.names)
            sim_obj.set_state(states[state_key])
        self.robot.SetDOFLimits(*states["dof_limits"])
        self.robot.SetDOFValues(states["dof_values"])

    @property
    def viewer(self):
        if not self.__viewer and trajoptpy.ViewerExists(self.env):
            self.__viewer = trajoptpy.GetViewer(self.env)
        return self.__viewer

    def _exclude_gripper_finger_collisions(self):
        if not self.robot:
            return
        cc = trajoptpy.GetCollisionChecker(self.env)
        for lr in 'lr':
            for flr in 'lr':
                finger_link_name = "%s_gripper_%s_finger_tip_link"%(lr,flr)
                finger_link = self.robot.GetLink(finger_link_name)
                for sim_obj in self.sim_objs:
                    for bt_obj in sim_obj.get_bullet_objects():
                        for link in bt_obj.GetKinBody().GetLinks():
                            cc.ExcludeCollisionPair(finger_link, link)

    def _include_gripper_finger_collisions(self):
        if not self.robot:
            return
        cc = trajoptpy.GetCollisionChecker(self.env)
        for lr in 'lr':
            for flr in 'lr':
                finger_link_name = "%s_gripper_%s_finger_tip_link"%(lr,flr)
                finger_link = self.robot.GetLink(finger_link_name)
                for sim_obj in self.sim_objs:
                    for bt_obj in sim_obj.get_bullet_objects():
                        for link in bt_obj.GetKinBody().GetLinks():
                            cc.IncludeCollisionPair(finger_link, link)

    @staticmethod
    def simulation_state_equal(s0, s1):
        if s0[0] != s1[0]:
            return False
        d0 = s0[1]
        d1 = s1[1]
        if not set(d0.keys()) == set(d1.keys()):
            return False
        for (k, v0) in d0.iteritems():
            v1 = d1[k]
            if not np.all(v0 == v1):
                return False
        return True

class DynamicSimulation(StaticSimulation):
    def __init__(self, env=None):
        super(DynamicSimulation, self).__init__(env=env)
        self.dyn_sim_objs = []
        self.bt_env = None
        self.bt_robot = None
        self.dyn_bt_objs = []

    def add_objects(self, sim_objs):
        static_sim_objs = [sim_obj for sim_obj in sim_objs if not sim_obj.dynamic]
        dyn_sim_objs = [sim_obj for sim_obj in sim_objs if sim_obj.dynamic]
        self._include_gripper_finger_collisions()
        # add static objects
        super(DynamicSimulation, self).add_objects(static_sim_objs, consider_finger_collisions=False)
        # add dynamic objects
        self._remove_bullet()
        for sim_obj in dyn_sim_objs:
            self.sim_objs.append(sim_obj)
            self.dyn_sim_objs.append(sim_obj)
        self._create_bullet()
        self._exclude_gripper_finger_collisions()

    def remove_objects(self, sim_objs):
        static_sim_objs = [sim_obj for sim_obj in sim_objs if not sim_obj.dynamic]
        dyn_sim_objs = [sim_obj for sim_obj in sim_objs if sim_obj.dynamic]
        self._include_gripper_finger_collisions()
        # remove static objects
        super(DynamicSimulation, self).remove_objects(static_sim_objs, consider_finger_collisions=False)
        # remove dynamic objects
        self._remove_bullet()
        for sim_obj in dyn_sim_objs:
            self.sim_objs.remove(sim_obj)
            self.dyn_sim_objs.remove(sim_obj)
        self._create_bullet()
        self._exclude_gripper_finger_collisions()

    def set_state(self, sim_state):
        """
        Defined such that execution1 and execution2 gives the same results if execution1 == execution2 in the following code execution:
        set_state(sim_state)
        execution1()
        set_state(sim_state)
        execution2()
        """
        self._include_gripper_finger_collisions()
        self._remove_bullet()
        self._create_bullet()
        self._exclude_gripper_finger_collisions()
        super(DynamicSimulation, self).set_state(sim_state)
        self.update()

    def update(self):
        self.bt_robot.UpdateBullet()
        self._update_rave()

    def step(self):
        self.bt_robot.UpdateBullet()
        self.bt_env.Step(.01, 200, .005)
        self._update_rave()

    def settle(self, max_steps=100, tol=.001, step_viewer=1):
        """Keep stepping until the dynamic objects doesn't move, up to some tolerance"""
        prev_trans = np.concatenate([np.asarray([link.GetTransform() for link in bt_obj.GetKinBody().GetLinks()])[:,:3,3] for bt_obj in self.dyn_bt_objs]) # translation part of all links of all dynamic objects
        for i in range(max_steps):
            self.bt_env.Step(.01, 200, .005)
            self._update_rave()
            if self.viewer and step_viewer!=0 and i%step_viewer==0:
                if self.viewer:
                    self.viewer.Step()
            if i % 10 == 0 and i != 0:
                curr_trans = np.concatenate([np.asarray([link.GetTransform() for link in bt_obj.GetKinBody().GetLinks()])[:,:3,3] for bt_obj in self.dyn_bt_objs])
                diff = np.sqrt(((curr_trans - prev_trans)**2).sum(axis=1))
                if diff.max() < tol:
                    print "settle tolerance reached"
                    break
                prev_trans = curr_trans
        self._update_rave()
        if self.viewer and step_viewer!=0:
            self.viewer.Step()

    def _create_bullet(self):
        # create bullet environment and dynamic objects in it
        dyn_obj_names = []
        for sim_obj in self.dyn_sim_objs:
            if not sim_obj.add_after:
                sim_obj.add_to_env(self)
                dyn_obj_names.extend(sim_obj.names)
        self.bt_env = bulletsimpy.BulletEnvironment(self.env, dyn_obj_names)
        self.bt_env.SetGravity([0, 0, -9.8])
        if self.robot:
            self.bt_robot = self.bt_env.GetObjectByName(self.robot.GetName())

        for sim_obj in self.dyn_sim_objs:
            if sim_obj.add_after:
                sim_obj.add_to_env(self)

        for sim_obj in self.dyn_sim_objs:
            self.dyn_bt_objs.extend(sim_obj.get_bullet_objects())

        self._update_rave()

    def _remove_bullet(self):
        # remove bullet environment and dynamic objects in it
        for sim_obj in self.dyn_sim_objs:
            sim_obj.remove_from_env()
        self.bt_env = None
        self.bt_robot = None
        self.dyn_bt_objs = []

    def _update_rave(self):
        for bt_obj in self.dyn_bt_objs:
            bt_obj.UpdateRave()
        self.env.UpdatePublishedBodies()

class DynamicSimulationRobotWorld(DynamicSimulation, RobotWorld):
    def __init__(self, env=None, T_w_k=None, range_k=2.):
        """
        T_w_k: world transform of the depth camera
        range_k: length of the rays. 2 meters by default
        """
        self.constraints = {"l": [], "r": []}
        self.constraints_links = {"l": [], "r": []}
        super(DynamicSimulationRobotWorld, self).__init__(env=env)
        self.T_w_k = T_w_k
        self.range_k = range_k

    def observe_cloud(self, observe_objs=None):
        if self.T_w_k is None:
            if self.robot is None:
                raise RuntimeError("Can't observe cloud when there is no robot")
            else:
                from rapprentice import berkeley_pr2
                self.T_w_k = berkeley_pr2.get_kinect_transform(self.robot)


        # camera's parameters
        cx = 320.-.5
        cy = 240.-.5
        f = 525. # focal length
        w = 640.
        h = 480.

        pixel_ij = np.array(np.meshgrid(np.arange(w, step=4), np.arange(h, step=4))).T.reshape((-1,2)) # all pixel positions
        rayTos = self.range_k * np.c_[(pixel_ij - np.array([cx, cy])) / f, np.ones(pixel_ij.shape[0])]
        rayFroms = np.zeros_like(rayTos)
        # transform the rays from the camera frame to the world frame
        rayTos = rayTos.dot(self.T_w_k[:3,:3].T) + self.T_w_k[:3,3]
        rayFroms = rayFroms.dot(self.T_w_k[:3,:3].T) + self.T_w_k[:3,3]

        measurements = {}

        cloud = []
        if observe_objs is None:
            observe_objs = self.dyn_sim_objs
        for sim_obj in observe_objs:
            for bt_obj in sim_obj.get_bullet_objects():
                ray_collisions = self.bt_env.RayTest(rayFroms, rayTos, bt_obj)


                for i, ray_collision in enumerate(ray_collisions):
                    key = (ray_collision.rayFrom[0], ray_collision.rayFrom[1], ray_collision.rayFrom[2],
                           ray_collision.rayTo[0], ray_collision.rayTo[1], ray_collision.rayTo[2])
                    d = np.linalg.norm(ray_collision.rayFrom - ray_collision.pt)
                    if key in measurements and d > measurements[key][0]:
                            continue
                    measurements[key] = (d, ray_collision.pt)
        cloud = np.zeros((len(measurements), 3))
        if not measurements:
            return np.array([0, 0, 0])
        for i, (_, pt) in enumerate(measurements.itervalues()):
            cloud[i, :] = pt
        # cloud = np.concatenate(cloud)

        # hack to filter out point below the top of the table. TODO: fix this hack
        table_sim_objs = [sim_obj for sim_obj in self.sim_objs if "table" in sim_obj.names]
        assert len(table_sim_objs) == 1
        table_sim_obj = table_sim_objs[0]
        table_height = table_sim_obj.translation[2] + table_sim_obj.extents[2]
        cloud = cloud[cloud[:,2] > table_height, :]
        return cloud

    def open_gripper(self, lr, target_val=None, step_viewer=1, max_vel=.02):
        #ipdb.set_trace()
        self._remove_constraints(lr)

        # generate gripper finger trajectory
        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = self.robot.GetDOFValues([joint_ind])[0]
        if target_val is None:
            target_val = sim_util.get_binary_gripper_angle(True)
        joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / max_vel))

        # execute gripper finger trajectory
        for val in joint_traj:
            self.robot.SetDOFValues([val], [joint_ind])
            self.step()
        if self.viewer and step_viewer:
            self.viewer.Step()

    def close_gripper(self, lr, step_viewer=10, max_vel=.001, close_dist_thresh=0.004, grab_dist_thresh=0.005):
        print 'CLOSING GRIPPER'
        # generate gripper finger trajectory
        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = self.robot.GetDOFValues([joint_ind])[0]
        print 'start_val: ', start_val
        # execute gripper finger trajectory
        dyn_bt_objs = [x for sim_obj in self.dyn_sim_objs for x in sim_obj.get_bullet_objects()]
        next_val = start_val
        i = 1
        while next_val:
            flr2finger_pts_grid = self._get_finger_pts_grid(lr)
            ray_froms, ray_tos = flr2finger_pts_grid['l'], flr2finger_pts_grid['r']

            # stop closing if any ray hits a dynamic object within a distance of close_dist_thresh from both sides
            next_vel = max_vel
            for bt_obj in dyn_bt_objs:
                from_to_ray_collisions = self.bt_env.RayTest(ray_froms, ray_tos, bt_obj)
                to_from_ray_collisions = self.bt_env.RayTest(ray_tos, ray_froms, bt_obj)
                rays_dists = np.inf * np.ones((len(ray_froms), 2))
                for rc in from_to_ray_collisions:
                    ray_id = np.argmin(np.apply_along_axis(np.linalg.norm, 1, ray_froms - rc.rayFrom))
                    rays_dists[ray_id,0] = np.linalg.norm(rc.pt - rc.rayFrom)
                for rc in to_from_ray_collisions:
                    ray_id = np.argmin(np.apply_along_axis(np.linalg.norm, 1, ray_tos - rc.rayFrom))
                    rays_dists[ray_id,1] = np.linalg.norm(rc.pt - rc.rayFrom)
                colliding_rays_inds = np.logical_and(rays_dists[:,0] != np.inf, rays_dists[:,1] != np.inf)
                if np.any(colliding_rays_inds):
                    rays_dists = rays_dists[colliding_rays_inds,:]
                    if np.any(np.logical_and(rays_dists[:,0] < close_dist_thresh, rays_dists[:,1] < close_dist_thresh)):
                        next_vel = 0
                    else:
                        next_vel = np.minimum(next_vel, np.min(rays_dists.sum(axis=1)))
            if next_vel == 0:
                break
            next_val = np.maximum(next_val - next_vel, 0)

            self.robot.SetDOFValues([next_val], [joint_ind])
            self.step()
            # if self.viewer and not i % step_viewer:
            #     self.viewer.Step()
            i += 1
        grasped_objs = set()
        handles = []
        # add constraints at the points where a ray hits a dynamic link within a distance of grab_dist_thresh
        for sim_obj in self.dyn_sim_objs:
            for bt_obj in sim_obj.get_bullet_objects():
                from_to_ray_collisions = self.bt_env.RayTest(ray_froms, ray_tos, bt_obj)
                to_from_ray_collisions = self.bt_env.RayTest(ray_tos, ray_froms, bt_obj)

                for i in range(ray_froms.shape[0]):
                    #handles.append(self.env.drawarrow(ray_froms[i,:], ray_tos[i,:]))
                    if step_viewer:
                        self.viewer.Step()
                ray_collisions = [rc for rcs in [from_to_ray_collisions, to_from_ray_collisions] for rc in rcs]
                #handles.append(self.env.plot3(np.asarray([x.pt for x in ray_collisions]), 2, (1,0,0)))
                n_hits = {}

                #self.viewer.Idle()
                #ipdb.set_trace()

                for rc in ray_collisions:
                    # if rc.link == bt_obj.GetKinBody().GetLink('rope_59'):
                    #     #handles.append(self.env.drawarrow(rc.rayFrom, rc.rayTo))
                    #     self.viewer.Step()
                    #     #ipdb.set_trace()
                    if np.linalg.norm(rc.pt - rc.rayFrom) < grab_dist_thresh:
                        if rc.link not in n_hits:
                            n_hits[rc.link] = 0
                        n_hits[rc.link] += 1
                        ## HACK so that we don't try to pick up bad objects
                        print rc.link
                        if (n_hits[rc.link] > 5 or
                            type(sim_obj) == CoilSimulationObject or type(sim_obj) == RopeSimulationObject):
                            grasped_objs.add(sim_obj)
                            link_tf = rc.link.GetTransform()
                            link_tf[:3,3] = rc.pt
                            self._add_constraints(lr, rc.link, link_tf)
        #ipdb.set_trace()
        if self.viewer and step_viewer:
            self.viewer.Step()
        return grasped_objs

    def execute_trajectory(self, full_traj, step_viewer=1, interactive=False, max_cart_vel_trans_traj=.05, sim_callback=None):
        """
        TODO: incorporate other parts of sim_full_traj_maybesim
        """
        if sim_callback is None:
            sim_callback = lambda i: self.step()

        traj, dof_inds = full_traj

    #     # clip finger joint angles to the binary gripper angles if necessary
    #     for lr in 'lr':
    #         joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
    #         if joint_ind in dof_inds:
    #             ind = dof_inds.index(joint_ind)
    #             traj[:,ind] = np.minimum(traj[:,ind], get_binary_gripper_angle(True))
    #             traj[:,ind] = np.maximum(traj[:,ind], get_binary_gripper_angle(False))

        # in simulation mode, we must make sure to gradually move to the new starting position
        self.robot.SetActiveDOFs(dof_inds)
        curr_vals = self.robot.GetActiveDOFValues()
        transition_traj = np.r_[[curr_vals], [traj[0]]]
        sim_util.unwrap_in_place(transition_traj, dof_inds=dof_inds)
        transition_traj = ropesim.retime_traj(self.robot, dof_inds, transition_traj, max_cart_vel=max_cart_vel_trans_traj)
        animate_traj.animate_traj(transition_traj, self.robot, restore=False, pause=interactive,
            callback=sim_callback, step_viewer=step_viewer if self.viewer else 0)

        traj[0] = transition_traj[-1]
        sim_util.unwrap_in_place(traj, dof_inds=dof_inds)
        traj = ropesim.retime_traj(self.robot, dof_inds, traj, max_cart_vel=.002) # make the trajectory slow enough for the simulation

        step_viewer*=5

        animate_traj.animate_traj(traj, self.robot, restore=False, pause=interactive,
            callback=sim_callback, step_viewer=step_viewer if self.viewer else 0)
        if self.viewer and step_viewer:
            self.viewer.Step()
        return True

    def _create_bullet(self):
        for lr in 'lr':
            if self.constraints[lr] or self.constraints_links[lr]:
                raise RuntimeError("Bullet environment can't be removed while the robot is grasping an object")
        super(DynamicSimulationRobotWorld, self)._create_bullet()

    def _remove_bullet(self):
        for lr in 'lr':
            if self.constraints[lr] or self.constraints_links[lr]:
                raise RuntimeError("Bullet environment can't be removed while the robot is grasping an object")
        super(DynamicSimulationRobotWorld, self)._remove_bullet()

    def _get_finger_pts_grid(self, lr, min_sample_dist=0.005):
        sample_grid = None
        flr2finger_pts_grid = {}
        for finger_lr in 'lr':
            world_from_finger = self.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,finger_lr)).GetTransform()
            finger_pts = world_from_finger[:3,3] + sim_util.get_finger_rel_pts(finger_lr).dot(world_from_finger[:3,:3].T)
            pt0 = finger_pts[0 if finger_lr == 'l' else 3][None,:]
            pt1 = finger_pts[1 if finger_lr == 'l' else 2][None,:]
            pt3 = finger_pts[3 if finger_lr == 'l' else 0][None,:]
            if sample_grid is None:
                num_sample_01 = np.round(np.linalg.norm(pt1 - pt0)/min_sample_dist)
                num_sample_03 = np.round(np.linalg.norm(pt3 - pt0)/min_sample_dist)
                sample_grid = np.array(np.meshgrid(np.linspace(0,1,num_sample_01), np.linspace(0,1,num_sample_03))).T.reshape((-1,2))
            flr2finger_pts_grid[finger_lr] = pt0 + sample_grid[:,0][:,None].dot(pt1 - pt0) + sample_grid[:,1][:,None].dot(pt3 - pt0)
        return flr2finger_pts_grid

    def _remove_constraints(self, lr, grab_link=None):
        """
        If grab_link is None, remove all constraints that attaches the lr gripper, else remove all constraints that attaches between the lr gripper and grab_link
        """
        #num_links_to_remove = len([link for link in self.constraints_links[lr] if link == grab_link])
        num_links_to_removed = 0
        links_size = len(self.constraints_links[lr])
        for (cnt, link) in zip(self.constraints[lr], self.constraints_links[lr]):
            if grab_link is None or link == grab_link:
                self.bt_env.RemoveConstraint(cnt)
                num_links_to_removed += 1
        # TODO: provide option to color the contrained links and save color before overriding it
        for link in self.constraints_links[lr]:
            if grab_link is None or link == grab_link:
                for geom in link.GetGeometries():
                    geom.SetDiffuseColor([1.,1.,1.])
        if grab_link is None:
            self.constraints[lr] = []
            self.constraints_links[lr] = []
        else:
            if grab_link in self.constraints_links[lr]:
                constraints_links_pairs = zip(*[(cnt, link) for (cnt, link) in zip(self.constraints[lr], self.constraints_links[lr]) if link != grab_link])
                if constraints_links_pairs:
                    constraints, constraints_links = constraints_links_pairs
                    self.constraints[lr] = list(constraints)
                    self.constraints_links[lr] = list(constraints_links)
                else:
                    self.constraints[lr] = []
                    self.constraints_links[lr] = []

    def _add_constraints(self, lr, grab_link, grab_tf=None):
        if grab_tf is None:
            grab_tf = grab_link.GetTransform()
        # TODO: provide option to color the contrained links and save color before overriding it
        for geom in grab_link.GetGeometries():
            geom.SetDiffuseColor([1.,0.,0.])
        for flr in 'lr':
            robot_link = self.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,flr))
            cnt = self.bt_env.AddConstraint({
                "type": "generic6dof",
                "params": {
                    "link_a": robot_link,
                    "link_b": grab_link,
                    "frame_in_a": np.linalg.inv(robot_link.GetTransform()).dot(grab_tf),
                    "frame_in_b": np.linalg.inv(grab_link.GetTransform()).dot(grab_tf),
                    "use_linear_reference_frame_a": False,
                    "stop_erp": .8,
                    "stop_cfm": .1,
                    "disable_collision_between_linked_bodies": True,
                }
            })
            self.constraints[lr].append(cnt)
            self.constraints_links[lr].append(grab_link)

class ClutterSimulationRobotWorld(DynamicSimulationRobotWorld):

    scale = 0.7

    TABLE_HEIGHT = .77
    BIG_DIM = scale * 0.05
    SMALL_DIM = scale * 0.03

    MAX_STEPS=5000

    def __init__(self, n_big, n_small, **kwargs):
        super(ClutterSimulationRobotWorld, self).__init__(**kwargs)
        self.coil = CoilSimulationObject("coil", np.eye(4), .05, sim_util.RopeParams())
        self.table = BoxSimulationObject("table",
                                         [1, 0, self.TABLE_HEIGHT-.1],
                                         [.85, 1, .1], dynamic=False)
        self.floor = BoxSimulationObject("floor",
                                         [1, 0, -self.TABLE_HEIGHT],
                                         [10, 10, .01], dynamic=False)
        big_dims = (self.BIG_DIM, self.BIG_DIM, self.BIG_DIM)
        self.container = ContainerSimulationObject()
        self.small_boxes = [BoxSimulationObject(
                "small_box_{}".format(i),
                (0, 0, 0),
                (self.SMALL_DIM, self.SMALL_DIM, self.SMALL_DIM),
                dynamic=True) for i in xrange(n_small)]
        self.big_boxes = [BoxSimulationObject(
                "big_box_{}".format(i),
                (0, 0, 0),
                (self.BIG_DIM, self.BIG_DIM, self.BIG_DIM),
                dynamic=True) for i in xrange(n_big)]
        robot = XmlSimulationObject("robots/pr2-beta-static.zae", dynamic=False)
        objs = [robot, self.table, self.container, self.coil, self.floor] + self.small_boxes + self.big_boxes
        self.add_objects(objs)
        self.color_container()
        self.reset_robot()
        self.reset_container()

    def color_container(self):
        for l in self.container.get_bullet_objects()[0].GetKinBody().GetLinks():
            for geom in l.GetGeometries():
                geom.SetDiffuseColor([1.,0.,0.])

    def set_state(self, *args):
        super(ClutterSimulationRobotWorld, self).set_state(*args)
        self.color_container()
        


    def reset_container(self):
        blt_obj = self.container.get_bullet_objects()[0]
        kinbdy = self.env.GetKinBody('container_kinbody')
        ## MAGIC
        T = openravepy.matrixFromAxisAngle(np.array([0, 0, -np.pi/6])).dot(kinbdy.GetTransform())
        T[0, 3] = .5
        T[1, 3] = -.3
        T[2, 3] = self.TABLE_HEIGHT + .01
        kinbdy.SetTransform(T)
        blt_obj.SetTransform(T)
        self.color_container()
        self.update()


    def init_coil(self, P):
        coil_args = self.coil.init_args
        new_args = (coil_args[0], P) + coil_args[2:]
        self.coil = CoilSimulationObject(*new_args, **self.coil.kwargs)
        self.add_objects([self.coil])

    def add_box(self, P, size='small'):
        P = (P[0, 3], P[1, 3], P[2, 3])

        if size == 'small':
            dim = self.SMALL_DIM
            other_boxes = self.small_boxes
        else:
            dim = self.BIG_DIM
            other_boxes = self.big_boxes

        name = size + '_box_{}'.format(len(other_boxes))
        new_box = BoxSimulationObject(
            name, P,
            (dim, dim, dim),
            dynamic=True)

        other_boxes.append(new_box)
        self.add_objects([new_box])

    def reset_robot(self):
        torso_idx, head_tilt_idx = (self.robot.GetJoint('torso_lift_joint').GetDOFIndex(),
                                    self.robot.GetJoint('head_tilt_joint').GetDOFIndex())
        _, torso_max = self.robot.GetJoint('torso_lift_joint').GetLimits()
        _, tilt_max = self.robot.GetJoint('head_tilt_joint').GetLimits()

        self.robot.SetDOFValues([torso_max, 0.8*tilt_max], [torso_idx, head_tilt_idx])
        sim_util.reset_arms_to_side(self)
        self.update()

    def initialize(self, state, step_viewer=False):
        self.reset_robot()

        dyn_objs = [self.coil] + self.small_boxes + self.big_boxes
        self.remove_objects(dyn_objs)
        self.small_boxes = []
        self.big_boxes = []

        for n in state:
            P = state[n]
            if n == 'coil':
                self.init_coil(P)
            else:
                size = 'small' if n.startswith('small') else 'big'
                self.add_box(P, size=size)
        self.reset_container()
        # self.viewer.Idle()

        self.settle(max_steps=self.MAX_STEPS, tol=1e-4, step_viewer=50)
        self.remove_cleared_objs()

    def settle(self, max_steps=None, tol=1e-3, step_viewer=10):
        if max_steps is None:
            max_steps = self.MAX_STEPS
        super(ClutterSimulationRobotWorld, self).settle(max_steps, tol, step_viewer)
        

    def observe_cloud(self, observe_objs=None):
        if observe_objs is None:
            observe_objs = self.observe_objs
        cld = super(ClutterSimulationRobotWorld, self).observe_cloud(observe_objs)
        assert cld.shape[1] == 3
        # Uncomment this to shift point clouds down or up.
        #cld[:, 2] = cld[:, 2] - .02
        return cld

    def remove_cleared_objs(self, debug=False):
        self.observe_objs = [self.container]
        c_x, c_y = self.container.get_footprint()
        # c_x += .05
        # c_y += .05
        wf_to_cf = np.linalg.inv(self.container.get_pose())

        removed_objs = []

        handles = []
        if debug:
            from rapprentice import plotting_openrave

            remove_region = np.zeros((20*20, 4))
            remove_region[:, 3] = 1
            grid_n = 20
            x_vals, y_vals = np.linspace(-c_x, c_x, grid_n), np.linspace(-c_y, c_y, grid_n)
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    remove_region[i + j * grid_n, :2] = [x, y]

            remove_region = self.container.get_pose().dot(remove_region.T)
            remove_region[2, :] = 1
            handles.append(self.env.plot3(remove_region[:3, :].T, 3, (1, 1, 0, 1)))

        rp_wf = self.coil.rope.GetControlPoints()
        rp_cf = wf_to_cf[:3, :3].dot(rp_wf.T) + wf_to_cf[:3, 3][:, None]
        coil_out = (np.abs(rp_cf[0]) > c_x) | (np.abs(rp_cf[1]) > c_y)
        if debug and np.any(coil_out):
            from rapprentice import plotting_openrave
            print_pts = rp_wf[coil_out]
            print_pts[:, 2] = print_pts[:, 2] + .3
            handles.append(self.env.plot3(print_pts, 3, (1, 1, 0, 1)))
            self.viewer.Idle()

        if not np.all(coil_out):
            self.observe_objs.append(self.coil)
        else:
            removed_objs.append(self.coil)

        for box in self.small_boxes + self.big_boxes:
            bt_cf = wf_to_cf.dot(box.get_bullet_objects()[0].GetTransform())
            box_x, box_y = np.abs(bt_cf[0, 3]), np.abs(bt_cf[1, 3])
            if box_x < c_x and box_y < c_y:
                self.observe_objs.append(box)
            else:
                removed_objs.append(box)
                if debug:
                    from rapprentice import plotting_openrave
                    print box_x, box_y
                    print self.container.extents
                    
                    print_pts = box.get_bullet_objects()[0].GetTransform()[:3, 3]
                    print_pts[2] = 1
                    print print_pts
                    handles.append(self.env.plot3(print_pts[None, :], 10, (1, 0, 0, 1)))
                    self.viewer.Idle()
                    
                    print "{} removed".format(box.name)
        return removed_objs

    def compute_reward(self, already_cleared_objs, grasped_objs):
        reward = -.1
        if self.container in grasped_objs:
            grasped_objs.remove(self.container)

        cur_cleared_objs = self.remove_cleared_objs()
        if len(grasped_objs) > 0:
            reward += 1
            if (self.coil not in grasped_objs + already_cleared_objs 
                and self.coil in cur_cleared_objs):
                # in case we pull the rope out with a box
                reward += 1

        return reward

class DynamicRopeSimulationRobotWorld(DynamicSimulationRobotWorld):

    def in_grasp_region(self,robot, lr, pt):
        tol = .00

        manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
        manip = robot.GetManipulator(manip_name)
        l_finger = robot.GetLink("%s_gripper_l_finger_tip_link"%lr)
        r_finger = robot.GetLink("%s_gripper_r_finger_tip_link"%lr)

        def transform(hmat, p):
            return hmat[:3,:3].dot(p) + hmat[:3,3]
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



    def close_gripper(self, lr, step_viewer=1, max_vel=.02, close_dist_thresh=0.004, grab_dist_thresh=0.005):
        print 'CLOSING GRIPPER'
        # generate gripper finger trajectory
        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = self.robot.GetDOFValues([joint_ind])[0]
        print 'start_val: ', start_val
        # execute gripper finger trajectory
        dyn_bt_objs = [bt_obj for sim_obj in self.dyn_sim_objs for bt_obj in sim_obj.get_bullet_objects()]
        next_val = start_val
        while next_val:
            flr2finger_pts_grid = self._get_finger_pts_grid(lr)
            ray_froms, ray_tos = flr2finger_pts_grid['l'], flr2finger_pts_grid['r']

            # stop closing if any ray hits a dynamic object within a distance of close_dist_thresh from both sides
            next_vel = max_vel
            for bt_obj in dyn_bt_objs:
                from_to_ray_collisions = self.bt_env.RayTest(ray_froms, ray_tos, bt_obj)
                to_from_ray_collisions = self.bt_env.RayTest(ray_tos, ray_froms, bt_obj)
                rays_dists = np.inf * np.ones((len(ray_froms), 2))
                for rc in from_to_ray_collisions:
                    ray_id = np.argmin(np.apply_along_axis(np.linalg.norm, 1, ray_froms - rc.rayFrom))
                    rays_dists[ray_id,0] = np.linalg.norm(rc.pt - rc.rayFrom)
                for rc in to_from_ray_collisions:
                    ray_id = np.argmin(np.apply_along_axis(np.linalg.norm, 1, ray_tos - rc.rayFrom))
                    rays_dists[ray_id,1] = np.linalg.norm(rc.pt - rc.rayFrom)
                colliding_rays_inds = np.logical_and(rays_dists[:,0] != np.inf, rays_dists[:,1] != np.inf)
                if np.any(colliding_rays_inds):
                    rays_dists = rays_dists[colliding_rays_inds,:]
                    if np.any(np.logical_and(rays_dists[:,0] < close_dist_thresh, rays_dists[:,1] < close_dist_thresh)):
                        next_vel = 0
                    else:
                        next_vel = np.minimum(next_vel, np.min(rays_dists.sum(axis=1)))
            if next_vel == 0:
                break
            next_val = np.maximum(next_val - next_vel, 0)

            self.robot.SetDOFValues([next_val], [joint_ind])
            self.step()
            if self.viewer and step_viewer:
                self.viewer.Step()


        rope = [bt_obj for sim_obj in self.dyn_sim_objs for bt_obj in sim_obj.get_bullet_objects()][0]
        nodes, ctl_pts = rope.GetNodes(), rope.GetControlPoints()

        graspable_nodes = np.array([self.in_grasp_region(self.robot, lr, n) for n in nodes])
        graspable_ctl_pts = np.array([self.in_grasp_region(self.robot, lr, n) for n in ctl_pts])
        graspable_inds = np.flatnonzero(np.logical_or(graspable_nodes, np.logical_or(graspable_ctl_pts[:-1], graspable_ctl_pts[1:])))
        print 'graspable inds for %s: %s' % (lr, str(graspable_inds))
        if len(graspable_inds) == 0:
            return False

        robot_link = self.robot.GetLink("%s_gripper_l_finger_tip_link"%lr)
        rope_links = rope.GetKinBody().GetLinks()
        for i_node in graspable_inds:
            i_cnt = i_node
            for geom in rope_links[i_cnt].GetGeometries():
                geom.SetDiffuseColor([1.,0.,0.])
            link = rope_links[i_cnt]
            self._add_constraints(lr, link, link.GetTransform())
        if step_viewer and self.viewer:
            self.viewer.Step()

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



