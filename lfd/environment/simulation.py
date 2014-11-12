from __future__ import division

import openravepy
import numpy as np
import importlib

import trajoptpy
import bulletsimpy


class StaticSimulation(object):
    def __init__(self, robot_type=None, env=None):
        if robot_type is None:
            from berkeley_PR2 import BerkeleyPR2
            robot_type = BerkeleyPR2
        if env is not None:
            self.env = env
        else:
            self.env = openravepy.Environment()
            self.env.StopSimulation()
        self.sim_objs = []
        self.robot = None
        self.robot_type = robot_type
        self.__viewer_cache = None
    
    def add_objects(self, objs_to_add):
        self.robot_type.pre_add_objects(self)
        for obj_to_add in objs_to_add:
            if obj_to_add.dynamic:
                raise RuntimeError("Dynamic object can't be added to StaticSimulation")
            else:
                self.sim_objs.append(obj_to_add)
                obj_to_add.add_to_env(self)
        if len(self.env.GetRobots()) > 1:
            raise NotImplementedError("Behavior for adding more than one robot has not been defined")
        self.robot = self.env.GetRobots()[-1]
        self.robot_type.post_add_objects(self)
    
    def remove_objects(self, objs_to_remove):
        self.robot_type.pre_remove_objects(self)
        for obj_to_remove in objs_to_remove:
            if obj_to_remove.dynamic:
                raise RuntimeError("Dynamic object can't be removed from StaticSimulation")
            else:
                self.sim_objs.remove(obj_to_remove)
                obj_to_remove.remove_from_env()
        if self.robot and self.robot not in self.env.GetRobots():
            raise NotImplementedError("Behavior for removing robots has not been defined")
        self.robot_type.post_remove_objects(self)
    
    def get_state(self):
        sim_objs_constructor_infos = [sim_obj._get_constructor_info() for sim_obj in self.sim_objs]
        
        states = {}
        for sim_obj in self.sim_objs:
            state_key = "".join(sim_obj.names)
            assert state_key not in states, "multiple sim_objs with same names"
            states[state_key] = sim_obj.get_state()
        states["dof_limits"] = np.asarray(self.robot.GetDOFLimits())
        states["dof_values"] = self.robot.GetDOFValues()
        
        sim_state = (sim_objs_constructor_infos, states)
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
        if not self.__viewer_cache and trajoptpy.ViewerExists(self.env):
            self.__viewer_cache = trajoptpy.GetViewer(self.env)
        return self.__viewer_cache

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
    def __init__(self, robot_type=None, env=None):
        if robot_type is None:
            from berkeley_PR2 import BerkeleyPR2
            robot_type = BerkeleyPR2
        super(DynamicSimulation, self).__init__(robot_type=robot_type, env=env)
        self.dyn_sim_objs = []
        self.bt_env = None
        self.bt_robot = None
        self.dyn_bt_objs = []
    
    def add_objects(self, sim_objs):
        static_sim_objs = [sim_obj for sim_obj in sim_objs if not sim_obj.dynamic]
        dyn_sim_objs = [sim_obj for sim_obj in sim_objs if sim_obj.dynamic]
        self.robot_type.pre_add_objects(self)
        # add static objects
        super(DynamicSimulation, self).add_objects(static_sim_objs)
        # add dynamic objects
        self._remove_bullet()
        for sim_obj in dyn_sim_objs:
            self.sim_objs.append(sim_obj)
            self.dyn_sim_objs.append(sim_obj)
        self._create_bullet()
        self.robot_type.post_add_objects(self)
    
    def remove_objects(self, sim_objs):
        static_sim_objs = [sim_obj for sim_obj in sim_objs if not sim_obj.dynamic]
        dyn_sim_objs = [sim_obj for sim_obj in sim_objs if sim_obj.dynamic]
        self.robot_type.pre_remove_objects(self)
        # remove static objects
        super(DynamicSimulation, self).remove_objects(static_sim_objs)
        # remove dynamic objects
        self._remove_bullet()
        for sim_obj in dyn_sim_objs:
            self.sim_objs.remove(sim_obj)
            self.dyn_sim_objs.remove(sim_obj)
        self._create_bullet()
        self.robot_type.post_remove_objects(self)
    
    def set_state(self, sim_state):
        """
        Defined such that execution1 and execution2 gives the same results
        if execution1 == execution2 in the following code:

        set_state(sim_state)
        execution1()
        set_state(sim_state)
        execution2()
        """
        self.robot_type.pre_set_state(self)
        self._remove_bullet()
        self._create_bullet()
        self.robot_type.post_set_state(self)
        super(DynamicSimulation, self).set_state(sim_state)
        self.update()

    def update(self):
        self.bt_robot.UpdateBullet()
        self._update_rave()
    
    def step(self):
        self.bt_robot.UpdateBullet()
        self.bt_env.Step(.01, 200, .005)
        self._update_rave()

    def get_translations(self):
        """return translation part of all links of all dynamic objects"""
        return np.concatenate(
            [np.asarray([link.GetTransform() for link in bt_obj.GetKinBody().GetLinks()])[:, :3, 3]
                for bt_obj in self.dyn_bt_objs]
        )

    def settle(self, max_steps=100, tol=.001, step_viewer=1):
        """Keep stepping until the dynamic objects doesn't move, up to some tolerance"""
        prev_trans = self.get_translations()
        for i in range(max_steps):
            self.bt_env.Step(.01, 200, .005)
            self._update_rave()
            if self.viewer and step_viewer != 0 and i % step_viewer == 0:
                if self.viewer:
                    self.viewer.Step()
            if i % 10 == 0 and i != 0:
                curr_trans = self.get_translations()
                diff = np.sqrt(((curr_trans - prev_trans)**2).sum(axis=1))
                if diff.max() < tol:
                    break
                prev_trans = curr_trans
        self._update_rave()
        if self.viewer and step_viewer != 0:
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



