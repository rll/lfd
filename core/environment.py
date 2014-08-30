from __future__ import division

import openravepy
import trajoptpy, bulletsimpy
from rapprentice import animate_traj, ropesim
import numpy as np
import demonstration, simulation_object, sim_util
import IPython as ipy

class LfdEnvironment(object):
    def __init__(self):
        self.ds_size = None
    
    def execute_augmented_trajectory(self, aug_traj, step_viewer=1, interactive=False):
        raise NotImplementedError

    def observe_scene(self):
        raise NotImplementedError

class SimulationEnvironment(LfdEnvironment):
    """
    TODO: cleanly separate the simulation components from the environment components
    """
    def __init__(self, sim_objs, downsample_size=0):
        self.downsample_size = downsample_size
        
        self.env = openravepy.Environment()
        self.env.StopSimulation()

        self.viewer = None
        self.constraints = {"l": [], "r": []}
        self.constraints_links = {"l": [], "r": []}
        
        self.sim_objs = sim_objs

        # create non-dynamic objects
        for sim_obj in self.sim_objs:
            if not sim_obj.dynamic:
                sim_obj.add_to_env(self)
        self.robot = self.env.GetRobots()[0] # one SimulationObject of the sim_objs needs to be a non-dynamic robot

        # create bullet environment and dynamic objects in it
        self.dyn_sim_objs = [sim_obj for sim_obj in self.sim_objs if sim_obj.dynamic]
        self.bt_env = None
        self.bt_robot = None   
        self.dyn_bt_objs = []
        self._create_bullet()

    def execute_augmented_trajectory(self, aug_traj, step_viewer=1, interactive=False):
        open_or_close_finger_traj = np.zeros(aug_traj.n_steps, dtype=bool)
        if aug_traj.lr2open_finger_traj is not None:
            for lr in aug_traj.lr2open_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2open_finger_traj[lr])
        if aug_traj.lr2close_finger_traj is not None:
            for lr in aug_traj.lr2close_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2close_finger_traj[lr])
        open_or_close_inds = np.where(open_or_close_finger_traj)[0]
        
        traj, dof_inds = aug_traj.get_full_traj(self.robot)
        ret = True
        lr2gripper_open = {'l':True, 'r':True}
        for (start_ind, end_ind) in zip(np.r_[0, open_or_close_inds], np.r_[open_or_close_inds+1, aug_traj.n_steps]):
            if aug_traj.lr2open_finger_traj is not None:
                for lr in aug_traj.lr2open_finger_traj.keys():
                    if aug_traj.lr2open_finger_traj[lr][start_ind]:
                        target_val = None
                        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                        if joint_ind in dof_inds:
                            target_val = traj[start_ind, dof_inds.index(joint_ind)]
                        self.open_gripper(lr, target_val=target_val, step_viewer=step_viewer)
                        lr2gripper_open[lr] = True
            if aug_traj.lr2close_finger_traj is not None:
                for lr in aug_traj.lr2close_finger_traj.keys():
                    if aug_traj.lr2close_finger_traj[lr][start_ind]:
                        self.close_gripper(lr, step_viewer=step_viewer)
                        lr2gripper_open[lr] = False
            # don't execute trajectory for finger joint if the corresponding gripper is closed
            active_inds = np.ones(len(dof_inds), dtype=bool)
            for lr in 'lr':
                if not lr2gripper_open[lr]:
                    joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                    if joint_ind in dof_inds:
                        active_inds[dof_inds.index(joint_ind)] = False
            miniseg_traj = traj[start_ind:end_ind, active_inds]
            miniseg_dof_inds = list(np.asarray(dof_inds)[active_inds])
            ret &= self.execute_trajectory((miniseg_traj, miniseg_dof_inds), step_viewer=step_viewer, interactive=interactive)
        return ret

    def observe_scene(self):
        T_w_k = self.robot.GetLink("wide_stereo_optical_frame").GetTransform() # TODO: use the actual kinect's frame

        pts = []
        for sim_obj in self.dyn_sim_objs:
            for bt_obj in sim_obj.get_bullet_objects():
                pts.append(self.raycast_cloud(T_w_k, bt_obj))
        full_cloud = np.concatenate(pts)

        return demonstration.SceneState(full_cloud, downsample_size=self.downsample_size)

    def raycast_cloud(self, T_w_k, obj, z=1.):
        """
        T_w_k: world transform of the depth camera
        obj: the BulletObject to do ray test onto
        z: length of the rays. 1 meter by default
        """
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
        
        return pts

    def open_gripper(self, lr, target_val=None, step_viewer=1):
        self._remove_constraints(lr)
        
        # generate gripper finger trajectory
        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = self.robot.GetDOFValues([joint_ind])[0]
        if target_val is None:
            target_val = sim_util.get_binary_gripper_angle(True)
        joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))

        # execute gripper finger trajectory
        for val in joint_traj:
            self.robot.SetDOFValues([val], [joint_ind])
            self.step()
        if step_viewer:
            self.viewer.Step()

    def close_gripper(self, lr, step_viewer=1, close_dist_thresh=0.0025, grab_dist_thresh=0.005):
        # generate gripper finger trajectory
        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = self.robot.GetDOFValues([joint_ind])[0]
        target_val = 0.0
        joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))

        # execute gripper finger trajectory
        dyn_bt_objs = [bt_obj for sim_obj in self.dyn_sim_objs for bt_obj in sim_obj.get_bullet_objects()]
        stop_closing = False
        for val in joint_traj:
            self.robot.SetDOFValues([val], [joint_ind])
            self.step()
            if step_viewer:
                self.viewer.Step()
            
            flr2finger_pts_grid = self._get_finger_pts_grid(lr)
            ray_froms, ray_tos = flr2finger_pts_grid['l'], flr2finger_pts_grid['r']

            # stop closing if any ray hits a dynamic object within a distance of close_dist_thresh from both sides
            for bt_obj in dyn_bt_objs:
                from_to_ray_collisions = self.bt_env.RayTest(ray_froms, ray_tos, bt_obj)
                for from_to_rc in from_to_ray_collisions:
                    if np.linalg.norm(from_to_rc.pt - from_to_rc.rayFrom) < close_dist_thresh:
                        to_from_ray_collisions = self.bt_env.RayTest(from_to_rc.rayTo[None,:], from_to_rc.rayFrom[None,:], bt_obj)
                        for to_from_rc in to_from_ray_collisions:
                            if np.linalg.norm(to_from_rc.pt - to_from_rc.rayFrom) < close_dist_thresh:
                                stop_closing = True
                                break
                if stop_closing:
                    break
            if stop_closing:
                break
        
        # add constraints at the points where a ray hits a dynamic link within a distance of grab_dist_thresh
        for bt_obj in dyn_bt_objs:
            from_to_ray_collisions = self.bt_env.RayTest(ray_froms, ray_tos, bt_obj)
            to_from_ray_collisions = self.bt_env.RayTest(ray_tos, ray_froms, bt_obj)
            ray_collisions = [rc for rcs in [from_to_ray_collisions, to_from_ray_collisions] for rc in rcs]
            for rc in ray_collisions:
                if np.linalg.norm(rc.pt - rc.rayFrom) < grab_dist_thresh:
                    link_tf = rc.link.GetTransform()
                    link_tf[:3,3] = rc.pt
                    self._add_constraints(lr, rc.link, link_tf)
                
        if step_viewer:
            self.viewer.Step()
    
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
        num_links_to_remove = len([link for link in self.constraints_links[lr] if link == grab_link])
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
    
    def execute_trajectory(self, full_traj, step_viewer=1, interactive=False, max_cart_vel_trans_traj=.05):
        """
        TODO: incorporate other parts of sim_full_traj_maybesim
        """
        def sim_callback(i):
            self.step()
        
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
            callback=sim_callback, step_viewer=step_viewer)
        
        traj[0] = transition_traj[-1]
        sim_util.unwrap_in_place(traj, dof_inds=dof_inds)
        traj = ropesim.retime_traj(self.robot, dof_inds, traj) # make the trajectory slow enough for the simulation
    
        animate_traj.animate_traj(traj, self.robot, restore=False, pause=interactive,
            callback=sim_callback, step_viewer=step_viewer)
        if step_viewer:
            self.viewer.Step()
        return True

    def add_object(self, sim_obj):
        if sim_obj.dynamic:
            self._remove_bullet()
            self.sim_objs.append(sim_obj)
            self.dyn_sim_objs.append(sim_obj)
            self._create_bullet()
        else:
            self.sim_objs.append(sim_obj)
            sim_obj.add_to_env(self)

    def remove_object(self, sim_obj):
        self.sim_objs.remove(sim_obj)
        if sim_obj.dynamic:
            self._remove_bullet()
            self.dyn_sim_objs.remove(sim_obj)
            self._create_bullet()
        else:
            if self.viewer:
                for bt_obj in sim_obj.get_bullet_objects():
                    self.viewer.RemoveKinBody(bt_obj.GetKinBody())
            sim_obj.remove_from_env()

    def step(self):
        self.bt_robot.UpdateBullet()
        self.bt_env.Step(.01, 200, .005)
        self._update_rave()

    def settle(self, max_steps=100, tol=.001, step_viewer=1):
        """Keep stepping until the dynamic objects doesn't move, up to some tolerance"""
        prev_trans = np.concatenate([np.asarray([link.GetTransform() for link in bt_obj.GetKinBody().GetLinks()])[:,:3,3] for bt_obj in self.dyn_bt_objs]) # translation part of all links of all dynamic objects
        for i in range(max_steps):
            self.bt_env.Step(.01, 200, .005)
            if self.viewer is not None and step_viewer!=0 and i%step_viewer==0:
                self._update_rave()
                self.viewer.Step()
            if i % 10 == 0 and i != 0:
                curr_trans = np.concatenate([np.asarray([link.GetTransform() for link in bt_obj.GetKinBody().GetLinks()])[:,:3,3] for bt_obj in self.dyn_bt_objs])
                diff = np.sqrt(((curr_trans - prev_trans)**2).sum(axis=1))
                if diff.max() < tol:
                    break
                prev_trans = curr_trans
        self._update_rave()
        if self.viewer is not None and step_viewer!=0:
            self.viewer.Step()

    def get_state(self):
        sim_obj_states = []
        for sim_obj in self.sim_objs:
            sim_obj_states.append(sim_obj.get_state())
        dof_limits = self.robot.GetDOFLimits()
        dof_values = self.robot.GetDOFValues()
        return (sim_obj_states, dof_limits, dof_values)

    def set_state(self, state):
        """
        Defined such that execution1 and execution2 gives the same results if execution1 == execution2 in the following code execution:
        set_state(state)
        execution1()
        set_state(state)
        execution2()
        """
        self._remove_bullet()
        self._create_bullet()
        sim_obj_states, dof_limits, dof_values = state
        for (sim_obj, sim_obj_state) in zip(self.sim_objs, sim_obj_states):
            sim_obj.set_state(sim_obj_state)
        self.robot.SetDOFLimits(*dof_limits)
        self.robot.SetDOFValues(dof_values)
        self._update_rave()

    def _remove_bullet(self):
        for lr in 'lr':
            assert not self.constraints[lr]
            assert not self.constraints_links[lr]
        
        self._include_gripper_finger_collisions()

        # remove bullet environment and dynamic objects in it
        for sim_obj in self.dyn_sim_objs:
            if self.viewer:
                for bt_obj in sim_obj.get_bullet_objects():
                    self.viewer.RemoveKinBody(bt_obj.GetKinBody())
            sim_obj.remove_from_env()
        self.bt_env = None
        self.bt_robot = None
        self.dyn_bt_objs = []

    def _create_bullet(self):
        for lr in 'lr':
            assert not self.constraints[lr]
            assert not self.constraints_links[lr]

        # create bullet environment and dynamic objects in it
        dyn_obj_names = []
        for sim_obj in self.dyn_sim_objs:
            if not sim_obj.add_after:
                sim_obj.add_to_env(self)
                dyn_obj_names.extend(sim_obj.names)

        self.bt_env = bulletsimpy.BulletEnvironment(self.env, dyn_obj_names)
        self.bt_env.SetGravity([0, 0, -9.8])
        self.bt_robot = self.bt_env.GetObjectByName(self.robot.GetName())

        for sim_obj in self.dyn_sim_objs:
            if sim_obj.add_after:
                sim_obj.add_to_env(self)
        
        for sim_obj in self.dyn_sim_objs:
            self.dyn_bt_objs.extend(sim_obj.get_bullet_objects())
        
        self._update_rave()
        self._exclude_gripper_finger_collisions()
        
    def _update_rave(self):
        for bt_obj in self.dyn_bt_objs:
            bt_obj.UpdateRave()
        self.env.UpdatePublishedBodies()
    
    def _exclude_gripper_finger_collisions(self):
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
        cc = trajoptpy.GetCollisionChecker(self.env)
        for lr in 'lr':
            for flr in 'lr':
                finger_link_name = "%s_gripper_%s_finger_tip_link"%(lr,flr)
                finger_link = self.robot.GetLink(finger_link_name)
                for sim_obj in self.sim_objs:
                    for bt_obj in sim_obj.get_bullet_objects():
                        for link in bt_obj.GetKinBody().GetLinks():
                            cc.IncludeCollisionPair(finger_link, link)

class GroundTruthRopeSimulationEnvironment(SimulationEnvironment):
    def __init__(self, sim_objs, upsample=0, upsample_rad=1, downsample_size=0):
        super(GroundTruthRopeSimulationEnvironment, self).__init__(sim_objs, downsample_size=downsample_size)
        self.upsample = upsample
        self.upsample_rad = upsample_rad
    
    def observe_scene(self):
        for sim_obj in self.sim_objs:
            if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                rope_sim_obj = sim_obj
                break
        return demonstration.GroundTruthRopeSceneState(rope_sim_obj.rope.GetControlPoints(), rope_sim_obj.rope_params.radius, upsample=self.upsample, upsample_rad=self.upsample_rad, downsample_size=self.downsample_size)

class RecordingSimulationEnvironment(SimulationEnvironment):
    def __init__(self, sim_objs, upsample=0, upsample_rad=1, downsample_size=0):
        super(RecordingSimulationEnvironment, self).__init__(sim_objs, downsample_size=downsample_size)
        self.upsample = upsample
        self.upsample_rad = upsample_rad
        self.cur_step_states = []

    def execute_augmented_trajectory(self, aug_traj, step_viewer=1, interactive=False):
        self.cur_step_states = []
        open_or_close_finger_traj = np.zeros(aug_traj.n_steps, dtype=bool)
        if aug_traj.lr2open_finger_traj is not None:
            for lr in aug_traj.lr2open_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2open_finger_traj[lr])
        if aug_traj.lr2close_finger_traj is not None:
            for lr in aug_traj.lr2close_finger_traj.keys():
                open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, aug_traj.lr2close_finger_traj[lr])
        open_or_close_inds = np.where(open_or_close_finger_traj)[0]
        
        traj, dof_inds = aug_traj.get_full_traj(self.robot)
        ret = True
        lr2gripper_open = {'l':True, 'r':True}
        for (start_ind, end_ind) in zip(np.r_[0, open_or_close_inds], np.r_[open_or_close_inds+1, aug_traj.n_steps]):
            if aug_traj.lr2open_finger_traj is not None:
                for lr in aug_traj.lr2open_finger_traj.keys():
                    if aug_traj.lr2open_finger_traj[lr][start_ind]:
                        target_val = None
                        joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                        if joint_ind in dof_inds:
                            target_val = traj[start_ind, dof_inds.index(joint_ind)]
                        self.open_gripper(lr, target_val=target_val, step_viewer=step_viewer)
                        lr2gripper_open[lr] = True
            if aug_traj.lr2close_finger_traj is not None:
                for lr in aug_traj.lr2close_finger_traj.keys():
                    if aug_traj.lr2close_finger_traj[lr][start_ind]:
                        self.close_gripper(lr, step_viewer=step_viewer)
                        lr2gripper_open[lr] = False
            # don't execute trajectory for finger joint if the corresponding gripper is closed
            active_inds = np.ones(len(dof_inds), dtype=bool)
            for lr in 'lr':
                if not lr2gripper_open[lr]:
                    joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
                    if joint_ind in dof_inds:
                        active_inds[dof_inds.index(joint_ind)] = False
            miniseg_traj = traj[start_ind:end_ind, active_inds]
            miniseg_dof_inds = list(np.asarray(dof_inds)[active_inds])
            ret &= self.execute_trajectory((miniseg_traj, miniseg_dof_inds), step_viewer=step_viewer, interactive=interactive)
        return ret

    def execute_trajectory(self, full_traj, step_viewer=1, interactive=False, max_cart_vel_trans_traj=.05):
        """
        TODO: incorporate other parts of sim_full_traj_maybesim
        """
        def sim_callback(i):
            for sim_obj in self.sim_objs:
                if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                    rope_sim_obj = sim_obj
                    break
            cur_state = demonstration.TimestepState(rope_sim_obj.rope.GetControlPoints(), self.robot, step=i)
            self.cur_step_states.append(cur_state)
            self.step()
        
        traj, dof_inds = full_traj

        # in simulation mode, we must make sure to gradually move to the new starting position
        self.robot.SetActiveDOFs(dof_inds)
        curr_vals = self.robot.GetActiveDOFValues()
        transition_traj = np.r_[[curr_vals], [traj[0]]]
        sim_util.unwrap_in_place(transition_traj, dof_inds=dof_inds)
        transition_traj = ropesim.retime_traj(self.robot, dof_inds, transition_traj, max_cart_vel=max_cart_vel_trans_traj)
        animate_traj.animate_traj(transition_traj, self.robot, restore=False, pause=interactive,
            callback=sim_callback, step_viewer=step_viewer)
        
        traj[0] = transition_traj[-1]
        sim_util.unwrap_in_place(traj, dof_inds=dof_inds)
        traj = ropesim.retime_traj(self.robot, dof_inds, traj) # make the trajectory slow enough for the simulation
    
        animate_traj.animate_traj(traj, self.robot, restore=False, pause=interactive,
            callback=sim_callback, step_viewer=step_viewer)
        if step_viewer:
            self.viewer.Step()
        return True

    def observe_scene(self):
        for sim_obj in self.sim_objs:
            if isinstance(sim_obj, simulation_object.RopeSimulationObject):
                rope_sim_obj = sim_obj
                break
        return demonstration.RecordingRopePositionsSceneState(rope_sim_obj.rope.GetControlPoints(), self.cur_step_states, rope_sim_obj.rope_params.radius, upsample=self.upsample, upsample_rad=self.upsample_rad, downsample_size=self.downsample_size)
