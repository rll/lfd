# Contains useful functions for PR2 rope tying simulation
# The purpose of this class is to eventually consolidate
# the various instantiations of do_task_eval.py

import h5py
import bulletsimpy
import openravepy, trajoptpy
import numpy as np
from numpy import asarray
import re
import IPython as ipy
import sys, os
import random

from rapprentice import animate_traj, ropesim, ros2rave, math_utils as mu, plotting_openrave, rope_initialization
from rapprentice.util import yellowprint
from constants import GRIPPER_OPEN_CLOSE_THRESH, ROPE_RADIUS, ROPE_ANG_STIFFNESS, ROPE_ANG_DAMPING, ROPE_LIN_DAMPING, \
    ROPE_ANG_LIMIT, ROPE_LIN_STOP_ERP, ROPE_MASS, ROPE_RADIUS_THICK, DS_SIZE, GRIPPER_MULT
PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)

class RopeState(object):
    def __init__(self, init_rope_nodes, rope_params, tfs=None):
        self.init_rope_nodes = init_rope_nodes
        self.rope_params = rope_params
        self.tfs = tfs

class RopeParams(object):
    def __init__(self):
        self.radius       = ROPE_RADIUS
        self.angStiffness = ROPE_ANG_STIFFNESS
        self.angDamping   = ROPE_ANG_DAMPING
        self.linDamping   = ROPE_LIN_DAMPING
        self.angLimit     = ROPE_ANG_LIMIT
        self.linStopErp   = ROPE_LIN_STOP_ERP
        self.mass         = ROPE_MASS

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)

class SceneState(object):
    ids = set()
    def __init__(self, cloud, rope_nodes, rope_state, id=None, color=None):
        self.cloud = cloud
        self.color = color
        self.rope_nodes = rope_nodes
        self.rope_state = rope_state
        if id is None:
            self.id = SceneState.get_unique_id()
        else:
            self.id = id
    
    @staticmethod
    def get_unique_id():
        id = len(SceneState.ids)
        assert id not in SceneState.ids
        SceneState.ids.add(id)
        return id

class SimulationEnv:
    def __init__(self, table_height, init_joint_names, init_joint_values, obstacles, dof_limits_factor):
        self.table_height = table_height
        self.init_joint_names = init_joint_names
        self.init_joint_values = init_joint_values
        self.obstacles = obstacles
        self.dof_limits_factor = dof_limits_factor
        self.robot = None
        self.env = None
        self.sim = None
        self.viewer = None
    
    def __getstate__(self):
        sim_env = SimulationEnv(self.table_height, self.init_joint_names, self.init_joint_values, self.obstacles, self.dof_limits_factor)
        return sim_env.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
    
    def initialize(self):
        self.env = openravepy.Environment()
        self.env.StopSimulation()
        self.env.Load("robots/pr2-beta-static.zae")
    #     self.env.Load("../data/misc/pr2-beta-static-decomposed-shoulder.zae")
        self.robot = self.env.GetRobots()[0]
    
        dof_inds = dof_inds_from_name(self.robot, '+'.join(self.init_joint_names))
        values, dof_inds = zip(*[(value, dof_ind) for value, dof_ind in zip(self.init_joint_values, dof_inds) if dof_ind != -1])
        self.robot.SetDOFValues(values, dof_inds)
 
        table_xml = make_table_xml(translation=[1, 0, self.table_height + (-.1 + .01)], extents=[.85, .85, .1])
    #     table_xml = make_table_xml(translation=[1-.3, 0, self.table_height + (-.1 + .01)], extents=[.85-.3, .85-.3, .1])
        self.env.LoadData(table_xml)
        if 'bookshelve' in self.obstacles:
            self.env.Load("data/bookshelves.env.xml")
        if 'boxes' in self.obstacles:
            self.env.LoadData(make_box_xml("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12]))
            self.env.LoadData(make_box_xml("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08]))
        if 'cylinders' in self.obstacles:
            self.env.LoadData(make_cylinder_xml("cylinder0", [.7,.43,table_height+(.01+.5)], .12, 1.))
            self.env.LoadData(make_cylinder_xml("cylinder1", [.7,-.43,table_height+(.01+.5)], .12, 1.))
            self.env.LoadData(make_cylinder_xml("cylinder2", [.4,.2,table_height+(.01+.65)], .06, .5))
            self.env.LoadData(make_cylinder_xml("cylinder3", [.4,-.2,table_height+(.01+.65)], .06, .5))
    
        cc = trajoptpy.GetCollisionChecker(self.env)
        for gripper_link in [link for link in self.robot.GetLinks() if 'gripper' in link.GetName()]:
            if self.env.GetKinBody('table'):
                cc.ExcludeCollisionPair(gripper_link, self.env.GetKinBody('table').GetLinks()[0])
    
        reset_arms_to_side(self)
    
        if self.dof_limits_factor != 1.0:
            assert 0 < self.dof_limits_factor and self.dof_limits_factor <= 1.0
            active_dof_indices = self.robot.GetActiveDOFIndices()
            active_dof_limits = self.robot.GetActiveDOFLimits()
            for lr in 'lr':
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                dof_inds = self.robot.GetManipulator(manip_name).GetArmIndices()
                limits = np.asarray(self.robot.GetDOFLimits(dof_inds))
                limits_mean = limits.mean(axis=0)
                limits_width = np.diff(limits, axis=0)
                new_limits = limits_mean + self.dof_limits_factor * np.r_[-limits_width/2.0, limits_width/2.0]
                for i, ind in enumerate(dof_inds):
                    active_dof_limits[0][active_dof_indices.tolist().index(ind)] = new_limits[0,i]
                    active_dof_limits[1][active_dof_indices.tolist().index(ind)] = new_limits[1,i]
            self.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])
    
#     def open_gripper(self, lr, animate=False):
#         """
#         TODO: generalize
#         """
#         self.sim.release_rope(lr)
# 
#         target_val = get_binary_gripper_angle(True)
#         
#         # execute gripper open/close trajectory
#         joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
#         start_val = self.robot.GetDOFValues([joint_ind])[0]
#         joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
#         for val in joint_traj:
#             self.robot.SetDOFValues([val], [joint_ind])
#             self.sim.step()
#         if animate:
#             self.viewer.Step()
# 
#     def close_gripper(self, lr, animate=False):
#         """
#         TODO: generalize
#         """
#         cc = trajoptpy.GetCollisionChecker(self.env)
#         for gripper_link in [link for link in self.robot.GetLinks() if 'gripper' in link.GetName()]:
#             for rope_link in self.sim.rope.GetKinBody().GetLinks():
#                 cc.IncludeCollisionPair(gripper_link, rope_link)
# 
#         target_val = 0.0
#         
#         # execute gripper open/close trajectory
#         joint_ind = self.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
#         start_val = self.robot.GetDOFValues([joint_ind])[0]
#         joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .01))
#         for val in joint_traj:
#             self.robot.SetDOFValues([val], [joint_ind])
#             self.sim.step()
#             if animate:
#                 self.viewer.Step()
# 
#             col_now = cc.BodyVsAll(self.robot)
#             col_all_link_name2body_link_name = {}
#             for cn in col_now:
#                 if cn.GetDistance() < 0.002:
#                     if cn.GetLinkAName() not in col_all_link_name2body_link_name:
#                         col_all_link_name2body_link_name[cn.GetLinkAName()] = []
#                     col_all_link_name2body_link_name[cn.GetLinkAName()].append(cn.GetLinkBName())
#             
#             grab_link_names = []
#             for all_link_name in col_all_link_name2body_link_name:
#                 if '%s_gripper_l_finger_tip_link'%lr in col_all_link_name2body_link_name[all_link_name] and '%s_gripper_r_finger_tip_link'%lr in col_all_link_name2body_link_name[all_link_name]:
#                     grab_link_names.append(all_link_name)
#             
#             if grab_link_names:
#                 break
#         
#         if grab_link_names:
#             link_name2link = {}
#             for body in self.env.GetBodies():
#                 for link in body.GetLinks():
#                     assert link.GetName() not in link_name2link
#                     link_name2link[link.GetName()] = link
# 
#             for grab_link_name in grab_link_names:
#                 robot_link = link_name2link["%s_gripper_l_finger_tip_link"%lr]
#                 grab_link = link_name2link[grab_link_name]
#                 for geom in grab_link.GetGeometries():
#                     geom.SetDiffuseColor([1.,0.,0.])
#                 cnt = self.sim.bt_env.AddConstraint({
#                     "type": "generic6dof",
#                     "params": {
#                         "link_a": robot_link,
#                         "link_b": grab_link,
#                         "frame_in_a": np.linalg.inv(robot_link.GetTransform()).dot(grab_link.GetTransform()),
#                         "frame_in_b": np.eye(4),
#                         "use_linear_reference_frame_a": False,
#                         "stop_erp": .8,
#                         "stop_cfm": .1,
#                         "disable_collision_between_linked_bodies": True,
#                     }
#                 })
#                 self.sim.constraints[lr].append(cnt)
#                 self.sim.constraints_links[lr].append(grab_link)
#             
#         for gripper_link in [link for link in self.robot.GetLinks() if 'gripper' in link.GetName()]:
#             for rope_link in self.sim.rope.GetKinBody().GetLinks():
#                 cc.ExcludeCollisionPair(gripper_link, rope_link)
#                 
#         if animate:
#             self.viewer.Step()
# 
#     def execute_trajectory(self, traj, animate=False, interactive=False):
#         open_or_close_finger_traj = np.zeros(traj.n_steps, dtype=bool)
#         if traj.lr2open_finger_traj is not None:
#             for lr in traj.lr2open_finger_traj.keys():
#                 open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, traj.lr2open_finger_traj[lr])
#         if traj.lr2close_finger_traj is not None:
#             for lr in traj.lr2close_finger_traj.keys():
#                 open_or_close_finger_traj = np.logical_or(open_or_close_finger_traj, traj.lr2close_finger_traj[lr])
#         open_or_close_inds = np.where(open_or_close_finger_traj)[0]
#         
#         full_traj = traj.get_full_traj(self.robot)
#         ret = True
#         for (start_ind, end_ind) in zip(np.r_[0, open_or_close_inds], np.r_[open_or_close_inds+1, traj.n_steps]):
#             if traj.lr2open_finger_traj is not None:
#                 for lr in traj.lr2open_finger_traj.keys():
#                     if traj.lr2open_finger_traj[lr][start_ind]:
#                         self.open_gripper(lr, animate=animate)
#             if traj.lr2close_finger_traj is not None:
#                 for lr in traj.lr2close_finger_traj.keys():
#                     if traj.lr2close_finger_traj[lr][start_ind]:
#                         self.close_gripper(lr, animate=animate)
#             ret &= sim_full_traj_maybesim(self, (full_traj[0][start_ind:end_ind,:], full_traj[1]), animate=animate, interactive=interactive)
#         return ret

    def set_rope_state(self, rope_state):
        replace_rope(rope_state.init_rope_nodes, self, rope_state.rope_params, restore=False)
        if rope_state.tfs is not None:
            set_rope_transforms(rope_state.tfs, self)
        self.sim.settle()
    
    def observe_scene(self, id=None, **kwargs):
        if kwargs['raycast']:
            new_cloud, endpoint_inds = self.sim.raycast_cloud(endpoints=3)
            if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
                return None
        else:
            new_cloud = self.sim.observe_cloud(upsample=kwargs['upsample'], upsample_rad=kwargs['upsample_rad'])
            endpoint_inds = np.zeros(len(new_cloud), dtype=bool) # for now, kwargs['raycast']=False is not compatible with kwargs['use_color']=True
        if kwargs['use_color']:
            new_cloud = color_cloud(new_cloud, endpoint_inds)
        if kwargs['downsample']:
            from rapprentice import clouds
            new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE)
        else:
            new_cloud_ds = new_cloud
        new_rope_nodes = self.sim.rope.GetControlPoints()
        new_rope_nodes= ropesim.observe_cloud(new_rope_nodes, self.sim.rope_params.radius, upsample=kwargs['upsample'])
        init_rope_nodes = self.sim.rope_pts
        rope_params = self.sim.rope_params
        tfs = get_rope_transforms(self)
        rope_state = RopeState(init_rope_nodes, rope_params, tfs)
        scene_state = SceneState(new_cloud_ds, new_rope_nodes, rope_state, id=id)
        return scene_state
    
def make_table_xml(translation, extents):
    xml = """
<LfdEnvironment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</LfdEnvironment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

def make_box_xml(name, translation, extents):
    xml = """
<LfdEnvironment>
  <KinBody name="%s">
    <Body type="dynamic" name="%s_link">
      <Translation>%f %f %f</Translation>
      <Geom type="box">
        <extents>%f %f %f</extents>
      </Geom>
    </Body>
  </KinBody>
</LfdEnvironment>
""" % (name, name, translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

def make_cylinder_xml(name, translation, radius, height):
    xml = """
<LfdEnvironment>
  <KinBody name="%s">
    <Body type="dynamic" name="%s_link">
      <Translation>%f %f %f</Translation>
      <Geom type="cylinder">
        <rotationaxis>1 0 0 90</rotationaxis>
        <radius>%f</radius>
        <height>%f</height>
      </Geom>
    </Body>
  </KinBody>
</LfdEnvironment>
""" % (name, name, translation[0], translation[1], translation[2], radius, height)
    return xml

def reset_arms_to_side(sim_env):
    sim_env.robot.SetDOFValues(PR2_L_POSTURES["side"],
                               sim_env.robot.GetManipulator("leftarm").GetArmIndices())
    #actionfile = None
    sim_env.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               sim_env.robot.GetManipulator("rightarm").GetArmIndices())
    open_angle = get_binary_gripper_angle(True)
    for lr in 'lr':
        joint_ind = sim_env.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = sim_env.robot.GetDOFValues([joint_ind])[0]
        sim_env.robot.SetDOFValues([open_angle], [joint_ind])

def arm_moved(joint_traj):    
    if len(joint_traj) < 2: return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()

def split_trajectory_by_gripper(seg_info):
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

    n_steps = len(lgrip)

    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= GRIPPER_OPEN_CLOSE_THRESH) & (lgrip[:-1] < GRIPPER_OPEN_CLOSE_THRESH))
    r_openings = np.flatnonzero((rgrip[1:] >= GRIPPER_OPEN_CLOSE_THRESH) & (rgrip[:-1] < GRIPPER_OPEN_CLOSE_THRESH))
    l_closings = np.flatnonzero((lgrip[1:] < GRIPPER_OPEN_CLOSE_THRESH) & (lgrip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))
    r_closings = np.flatnonzero((rgrip[1:] < GRIPPER_OPEN_CLOSE_THRESH) & (rgrip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def split_trajectory_by_lr_gripper(seg_info, lr):
    grip = asarray(seg_info["%s_gripper_joint"%lr])

    n_steps = len(grip)

    # indices BEFORE transition occurs
    openings = np.flatnonzero((grip[1:] >= GRIPPER_OPEN_CLOSE_THRESH) & (grip[:-1] < GRIPPER_OPEN_CLOSE_THRESH))
    closings = np.flatnonzero((grip[1:] < GRIPPER_OPEN_CLOSE_THRESH) & (grip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))

    before_transitions = np.r_[openings, closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def get_opening_closing_inds(finger_traj):
    GRIPPER_OPEN_CLOSE_THRESH = 0.01 # TODO in constants
    
    mult = 5.0
    GRIPPER_L_FINGER_OPEN_CLOSE_THRESH = mult * GRIPPER_OPEN_CLOSE_THRESH

    # indices BEFORE transition occurs
    opening_inds = np.flatnonzero((finger_traj[1:] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    closing_inds = np.flatnonzero((finger_traj[1:] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    
    return opening_inds, closing_inds

def gripper_joint2gripper_l_finger_joint_values(gripper_joint_vals):
    """
    Only the %s_gripper_l_finger_joint%lr can be controlled (this is the joint returned by robot.GetManipulator({"l":"leftarm", "r":"rightarm"}[lr]).GetGripperIndices())
    The rest of the gripper joints (like %s_gripper_joint%lr) are mimiced and cannot be controlled directly
    """
    gripper_l_finger_joint_vals = GRIPPER_MULT * gripper_joint_vals
    return gripper_l_finger_joint_vals

def gripper_l_finger_joint2gripper_joint_values(gripper_l_finger_joint_vals):
    """
    Only the %s_gripper_l_finger_joint%lr can be controlled (this is the joint returned by robot.GetManipulator({"l":"leftarm", "r":"rightarm"}[lr]).GetGripperIndices())
    The rest of the gripper joints (like %s_gripper_joint%lr) are mimiced and cannot be controlled directly
    """
    gripper_joint_vals = gripper_l_finger_joint_vals / GRIPPER_MULT
    return gripper_joint_vals


def binarize_gripper(angle):
    return angle > GRIPPER_OPEN_CLOSE_THRESH

def get_binary_gripper_angle(open):
    mult = 5
    open_angle = .08 * mult
    closed_angle = .015 * mult
    return open_angle if open else closed_angle

def set_gripper_maybesim(sim_env, lr, is_open, prev_is_open, animate=False):
    target_val = get_binary_gripper_angle(is_open)
    
    # release constraints if necessary
    if is_open and not prev_is_open:
        sim_env.sim.release_rope(lr)
        print "DONE RELEASING"

    # execute gripper open/close trajectory
    joint_ind = sim_env.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
    start_val = sim_env.robot.GetDOFValues([joint_ind])[0]
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        sim_env.robot.SetDOFValues([val], [joint_ind])
        sim_env.sim.step()
#         if args.animation:
#                sim_env.viewer.Step()
#             if args.interactive: sim_env.viewer.Idle()
    # add constraints if necessary
    if animate:
        sim_env.viewer.Step()
    if not is_open and prev_is_open:
        if not sim_env.sim.grab_rope(lr):
            return False

    return True

def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2,4,6]:
        traj[:,i] = np.unwrap(traj[:,i])
    return traj

def unwrap_in_place(t, dof_inds=None):
    if dof_inds is not None:
        unwrap_inds = [dof_inds.index(dof_ind) for dof_ind in [17, 19, 21, 29, 31, 33] if dof_ind in dof_inds]
        for i in unwrap_inds:
            t[:,i] = np.unwrap(t[:,i])
    else:
        # TODO: do something smarter than just checking shape[1]
        if t.shape[1] == 7:
            unwrap_arm_traj_in_place(t)
        elif t.shape[1] == 14:
            unwrap_arm_traj_in_place(t[:,:7])
            unwrap_arm_traj_in_place(t[:,7:])
        else:
            raise NotImplementedError

def dof_inds_from_name(robot, name):
    dof_inds = []
    for component in name.split('+'):
        if robot.GetManipulator(component) is not None:
            dof_inds.extend(robot.GetManipulator(component).GetArmIndices())
        elif robot.GetJoint(component) is not None:
            dof_inds.append(robot.GetJoint(component).GetDOFIndex())
        else:
            raise NotImplementedError, "error in reading manip description: %s must be a manipulator or link"%component 
    return dof_inds

def sim_traj_maybesim(sim_env, lr2traj, animate=False, interactive=False, max_cart_vel_trans_traj=.05):
    full_traj = get_full_traj(sim_env, lr2traj)
    return sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive, max_cart_vel_trans_traj=max_cart_vel_trans_traj)

def sim_full_traj_maybesim(sim_env, full_traj, animate=False, interactive=False, max_cart_vel_trans_traj=.05):
    def sim_callback(i):
        sim_env.sim.step()

    animate_speed = 20 if animate else 0

    traj, dof_inds = full_traj
    
    # clip finger joint angles to the binary gripper angles if necessary
    for lr in 'lr':
        joint_ind = sim_env.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        if joint_ind in dof_inds:
            ind = dof_inds.index(joint_ind)
            traj[:,ind] = np.minimum(traj[:,ind], get_binary_gripper_angle(True))
            traj[:,ind] = np.maximum(traj[:,ind], get_binary_gripper_angle(False))
    
    # in simulation mode, we must make sure to gradually move to the new starting position
    sim_env.robot.SetActiveDOFs(dof_inds)
    curr_vals = sim_env.robot.GetActiveDOFValues()
    transition_traj = np.r_[[curr_vals], [traj[0]]]
    unwrap_in_place(transition_traj, dof_inds=dof_inds)
    transition_traj = ropesim.retime_traj(sim_env.robot, dof_inds, transition_traj, max_cart_vel=max_cart_vel_trans_traj)
    animate_traj.animate_traj(transition_traj, sim_env.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    
    traj[0] = transition_traj[-1]
    unwrap_in_place(traj, dof_inds=dof_inds)
    traj = ropesim.retime_traj(sim_env.robot, dof_inds, traj) # make the trajectory slow enough for the simulation

    valid_inds = grippers_exceed_rope_length(sim_env, (traj, dof_inds), 0.05)
    min_gripper_dist = [np.inf] # minimum distance between gripper when the rope capsules are too far apart
    def is_rope_pulled_too_tight(i_step):
        if valid_inds is None or valid_inds[i_step]: # gripper is not holding the rope or the grippers are not that far apart
            return True
        rope = sim_env.sim.rope
        trans = rope.GetTranslations()
        hhs = rope.GetHalfHeights()
        rots = rope.GetRotations()
        fwd_pts = (trans + hhs[:,None]*rots[:,:3,0])
        bkwd_pts = (trans - hhs[:,None]*rots[:,:3,0])
        pts_dists = np.apply_along_axis(np.linalg.norm, 1, fwd_pts[:-1] - bkwd_pts[1:])[:,None] # these should all be zero if the rope constraints are satisfied
        if np.any(pts_dists > sim_env.sim.rope_params.radius):
            if i_step == 0:
                return True
            ee_trajs = {}
            for lr in 'lr':
                ee_trajs[lr] = get_ee_traj(sim_env, lr, (traj[i_step-1:i_step+1], dof_inds), ee_link_name_fmt="%s_gripper_l_finger_tip_link")
            min_gripper_dist[0] = min(min_gripper_dist[0], np.linalg.norm(ee_trajs['r'][0,:3,3] - ee_trajs['l'][0,:3,3]))
            grippers_moved_closer = np.linalg.norm(ee_trajs['r'][1,:3,3] - ee_trajs['l'][1,:3,3]) < min_gripper_dist[0]
            return grippers_moved_closer
        return True
    animate_traj.animate_traj(traj, sim_env.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed, execute_step_cond=is_rope_pulled_too_tight)
    if min_gripper_dist[0] != np.inf:
        yellowprint("Some steps of the trajectory were not executed because the gripper was pulling the rope too tight.")
    if animate:
        sim_env.viewer.Step()
    return True

def get_full_traj(robot, lr2arm_traj, lr2finger_traj = {}):
    """
    A full trajectory is a tuple of a trajectory (np matrix) and dof indices (list)
    """
    trajs = []
    dof_inds = []
    if len(lr2arm_traj) > 0:
        for (lr, arm_traj) in lr2arm_traj.items():
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            trajs.append(arm_traj)
            dof_inds.extend(robot.GetManipulator(manip_name).GetArmIndices())
    if len(lr2finger_traj) > 0:
        for (lr, finger_traj) in lr2finger_traj.items():
            trajs.append(finger_traj)
            dof_inds.append(robot.GetJointIndex("%s_gripper_l_finger_joint"%lr))
    if len(trajs) > 0:
        full_traj = (np.concatenate(trajs, axis=1), dof_inds)
    else:
        full_traj = (np.zeros((0,0)), [])
    return full_traj

def merge_full_trajs(full_trajs):
    trajs = []
    dof_inds = []
    if len(full_trajs) > 0:
        for full_traj in full_trajs:
            trajs.append(full_traj[0])
            dof_inds.extend(full_traj[1])
        n_steps = np.max([len(traj) for traj in trajs])
        for i, traj in enumerate(trajs):
            if len(traj) < n_steps:
                trajs[i] = np.r_[traj, np.tile(traj[-1], (n_steps-len(traj),1))]
        full_traj = (np.concatenate(trajs, axis=1), dof_inds)
    else:
        full_traj = (np.zeros((0,0)), [])
    return full_traj

def get_ee_traj(robot, lr, arm_traj_or_full_traj, ee_link_name_fmt="%s_gripper_tool_frame"):
    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    ee_link_name = ee_link_name_fmt%lr
    ee_link = robot.GetLink(ee_link_name)
    if type(arm_traj_or_full_traj) == tuple: # it is a full_traj
        full_traj = arm_traj_or_full_traj
        traj = full_traj[0]
        dof_inds = full_traj[1]
    else:
        arm_traj = arm_traj_or_full_traj
        traj = arm_traj
        dof_inds = robot.GetManipulator(manip_name).GetArmIndices()
    ee_traj = []
    with openravepy.RobotStateSaver(robot):
        for i_step in range(traj.shape[0]):
            robot.SetDOFValues(traj[i_step], dof_inds)
            ee_traj.append(ee_link.GetTransform())
    return np.array(ee_traj)

def get_finger_rel_pts(finger_lr):
    left_rel_pts = np.array([[.027,-.016, .01], [-.002,-.016, .01], 
                             [-.002,-.016,-.01], [.027,-.016,-.01]])
    if finger_lr == 'l':
        return left_rel_pts
    else:
        rot_x_180 = np.diag([1,-1,-1])
        return left_rel_pts.dot(rot_x_180.T)

def get_finger_pts_traj(robot, lr, full_traj_or_ee_finger_traj):
    """
    ee_traj = sim_util.get_ee_traj(robot, lr, arm_traj)
    flr2finger_pts_traj1 = get_finger_pts_traj(robot, lr, (ee_traj, finger_traj))
    
    full_traj = sim_util.get_full_traj(robot, {lr:arm_traj}, {lr:finger_traj})
    flr2finger_pts_traj2 = get_finger_pts_traj(robot, lr, full_traj)
    """
    flr2finger_pts_traj = {}
    assert type(full_traj_or_ee_finger_traj) == tuple
    if full_traj_or_ee_finger_traj[0].ndim == 3:
        ee_traj, finger_traj = full_traj_or_ee_finger_traj
        assert len(ee_traj) == len(finger_traj)
        for finger_lr in 'lr':
            gripper_full_traj = get_full_traj(robot, {}, {lr:finger_traj})
            rel_ee_traj = get_ee_traj(robot, lr, gripper_full_traj)
            rel_finger_traj = get_ee_traj(robot, lr, gripper_full_traj, ee_link_name_fmt="%s"+"_gripper_%s_finger_tip_link"%finger_lr)
            
            flr2finger_pts_traj[finger_lr] = []
            for (world_from_ee, world_from_rel_ee, world_from_rel_finger) in zip(ee_traj, rel_ee_traj, rel_finger_traj):
                ee_from_finger = mu.invertHmat(world_from_rel_ee).dot(world_from_rel_finger)
                world_from_finger = world_from_ee.dot(ee_from_finger)
                finger_pts = world_from_finger[:3,3] + get_finger_rel_pts(finger_lr).dot(world_from_finger[:3,:3].T)
                flr2finger_pts_traj[finger_lr].append(finger_pts)
            flr2finger_pts_traj[finger_lr] = np.asarray(flr2finger_pts_traj[finger_lr])
    else:
        full_traj = full_traj_or_ee_finger_traj
        for finger_lr in 'lr':
            finger_traj = get_ee_traj(robot, lr, full_traj, ee_link_name_fmt="%s"+"_gripper_%s_finger_tip_link"%finger_lr)
            flr2finger_pts_traj[finger_lr] = []
            for world_from_finger in finger_traj:
                flr2finger_pts_traj[finger_lr].append(world_from_finger[:3,3] + get_finger_rel_pts(finger_lr).dot(world_from_finger[:3,:3].T))
            flr2finger_pts_traj[finger_lr] = np.asarray(flr2finger_pts_traj[finger_lr])
    return flr2finger_pts_traj

def grippers_exceed_rope_length(sim_env, full_traj, thresh):
    """
    Let min_length be the minimun length of the rope between the parts being held by the left and right gripper.
    This function returns a mask of the trajectory steps in which the distance between the grippers doesn't exceed min_length-thresh.
    If not both of the grippers are holding the rope, this function return None.
    """
    if sim_env.sim.constraints['l'] and sim_env.sim.constraints['r']:
        ee_trajs = {}
        for lr in 'lr':
            ee_trajs[lr] = get_ee_traj(sim_env.robot, lr, full_traj, ee_link_name_fmt="%s_gripper_l_finger_tip_link")
        min_length = np.inf
        hs = sim_env.sim.rope.GetHalfHeights()
        rope_links = sim_env.sim.rope.GetKinBody().GetLinks()
        for l_rope_link in sim_env.sim.constraints_links['l']:
            if l_rope_link not in rope_links:
                continue
            for r_rope_link in sim_env.sim.constraints_links['r']:
                if r_rope_link not in rope_links:
                    continue
                i_cnt_l = rope_links.index(l_rope_link)
                i_cnt_r = rope_links.index(r_rope_link)
                if i_cnt_l > i_cnt_r:
                    i_cnt_l, i_cnt_r = i_cnt_r, i_cnt_l
                min_length = min(min_length, 2*hs[i_cnt_l+1:i_cnt_r].sum() + hs[i_cnt_l] + hs[i_cnt_r])
        valid_inds = np.apply_along_axis(np.linalg.norm, 1, (ee_trajs['r'][:,:3,3] - ee_trajs['l'][:,:3,3])) < min_length - thresh
        return valid_inds
    else:
        return None

def remove_tight_rope_pull(sim_env, full_traj):
    if sim_env.sim.constraints['l'] and sim_env.sim.constraints['r']:
        ee_trajs = {}
        for lr in 'lr':
            ee_trajs[lr] = get_ee_traj(sim_env, lr, full_traj, ee_link_name_fmt="%s_gripper_l_finger_tip_link")
        min_length = np.inf
        hs = sim_env.sim.rope.GetHalfHeights()
        rope_links = sim_env.sim.rope.GetKinBody().GetLinks()
        for l_rope_link in sim_env.sim.constraints_links['l']:
            if l_rope_link not in rope_links:
                continue
            for r_rope_link in sim_env.sim.constraints_links['r']:
                if r_rope_link not in rope_links:
                    continue
                i_cnt_l = rope_links.index(l_rope_link)
                i_cnt_r = rope_links.index(r_rope_link)
                if i_cnt_l > i_cnt_r:
                    i_cnt_l, i_cnt_r = i_cnt_r, i_cnt_l
                min_length = min(min_length, 2*hs[i_cnt_l+1:i_cnt_r].sum() + hs[i_cnt_l] + hs[i_cnt_r])
        valid_inds = np.apply_along_axis(np.linalg.norm, 1, (ee_trajs['r'][:,:3,3] - ee_trajs['l'][:,:3,3])) < min_length - 0.02
        if not np.all(valid_inds):
            full_traj = (full_traj[0][valid_inds,:], full_traj[1])
            yellowprint("The grippers of the trajectory goes too far apart. Some steps of the trajectory are being removed.")
    return full_traj

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz'][:,:3], seg_name

def load_fake_data_segment(demofile, fake_data_segment, fake_data_transform):
    fake_seg = demofile[fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])[:,:3]
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])
    hmat[:3,3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
    joint_names = asarray(fake_seg["joint_states"]["name"])
    joint_values = asarray(fake_seg["joint_states"]["position"][0])
    return new_xyz, joint_names, joint_values

def unif_resample(traj, max_diff, wt = None):        
    """
    Resample a trajectory so steps have same length in joint space    
    """
    import scipy.interpolate as si
    tol = .005
    if wt is not None: 
        wt = np.atleast_2d(wt)
        traj = traj*wt
        
    dl = mu.norms(traj[1:] - traj[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1: return traj, np.arange(len(traj))
    
    nsteps = max(int(np.ceil(float(l[-1])/max_diff)), 2)
    newl = np.linspace(0,l[-1],nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps,ncols)) 
    for istart in xrange(0, traj.shape[1], colstep):
        (tck,_) = si.splprep(traj[goodinds, istart:istart+colstep].T,k=deg,s = tol**2*len(traj),u=l[goodinds])
        traj_rs[:,istart:istart+colstep] = np.array(si.splev(newl,tck)).T
    if wt is not None: traj_rs = traj_rs/wt

    newt = np.interp(newl, l, np.arange(len(traj)))

    return traj_rs, newt

def get_rope_transforms(sim_env):
    return (sim_env.sim.rope.GetTranslations(), sim_env.sim.rope.GetRotations())    

def replace_rope(new_rope, sim_env, rope_params, restore=False):
    """
    restore indicates if this function is being called to restore an existing rope, in which case the color of the rope is saved and restored
    """
    if sim_env.sim:
        for lr in 'lr':
            sim_env.sim.release_rope(lr)
    rope_kin_body = sim_env.env.GetKinBody('rope')
    geom_colors = []
    if restore:
        assert rope_kin_body is not None # the rope should already exist when restore is happening
        for link in rope_kin_body.GetLinks():
            for geom in link.GetGeometries():
                geom_colors.append(geom.GetDiffuseColor())
    if rope_kin_body:
        if sim_env.viewer:
            sim_env.viewer.RemoveKinBody(rope_kin_body)
    if sim_env.sim:
        del sim_env.sim
    sim_env.sim = ropesim.Simulation(sim_env.env, sim_env.robot, rope_params)
    sim_env.sim.create(new_rope)
    rope_kin_body = sim_env.env.GetKinBody('rope')
    if restore:
        assert len(geom_colors) == len(rope_kin_body.GetLinks()) # the old and new rope should have the same number of links when restore is happening
        for link in rope_kin_body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetDiffuseColor(geom_colors.pop(0))

def set_rope_transforms(tfs, sim_env):
    sim_env.sim.rope.SetTranslations(tfs[0])
    sim_env.sim.rope.SetRotations(tfs[1])

def get_rope_params(params_id):
    rope_params = bulletsimpy.CapsuleRopeParams()
    rope_params.radius       = ROPE_RADIUS
    rope_params.angStiffness = ROPE_ANG_STIFFNESS
    rope_params.angDamping   = ROPE_ANG_DAMPING
    rope_params.linDamping   = ROPE_LIN_DAMPING
    rope_params.angLimit     = ROPE_ANG_LIMIT
    rope_params.linStopErp   = ROPE_LIN_STOP_ERP
    rope_params.mass         = ROPE_MASS

    if params_id == 'default':
        pass
    elif params_id == 'thick':
        rope_params.radius = ROPE_RADIUS_THICK        
    elif params_id.startswith('stiffness'):
        try:
            stiffness = float(re.search(r'stiffness(.*)', params_id).group(1))
        except:
            raise RuntimeError("Invalid rope parameter id")
        rope_params.angStiffness = stiffness
    else:
        raise RuntimeError("Invalid rope parameter id")
    return rope_params

def tpsrpm_plot_cb(sim_env, x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(sim_env.env.plot3(ypred_nd, 3, (0,1,0,1)))
    handles.extend(plotting_openrave.draw_grid(sim_env.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if sim_env.viewer:
        sim_env.viewer.Step()

def draw_grid(sim_env, old_xyz, f, color = (1,1,0,1)):
    grid_means = .5 * (old_xyz.max(axis=0) + old_xyz.min(axis=0))
    grid_mins = grid_means - (old_xyz.max(axis=0) - old_xyz.min(axis=0))
    grid_maxs = grid_means + (old_xyz.max(axis=0) - old_xyz.min(axis=0))
    return plotting_openrave.draw_grid(sim_env.env, f.transform_points, grid_mins, grid_maxs, 
                                       xres = .1, yres = .1, zres = .04, color = color)

def draw_axis(sim_env, hmat):
    handles = []
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,0]/10.0, 0.005, (1,0,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,1]/10.0, 0.005, (0,1,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,2]/10.0, 0.005, (0,0,1,1)))
    return handles

def draw_finger_pts_traj(sim_env, flr2finger_pts_traj, color):
    handles = []
    for finger_lr, pts_traj in flr2finger_pts_traj.items():
        for pts in pts_traj:
            handles.append(sim_env.env.drawlinestrip(np.r_[pts, pts[0][None,:]], 1, color))
    return handles

def one_l_print(string, pad=20):
    for _ in range(pad): string += ' '
    string += '\r'
    sys.stdout.write(string)
    sys.stdout.flush()

class suppress_stdout(object):
    '''
    A context manager for doing a "deep suppression" of stdout in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    '''
    def __init__(self):
        # Open a null file
        self.null_fds =  os.open(os.devnull,os.O_RDWR)
        # Save the actual stdout file descriptor
        self.save_fds = os.dup(1)

    def __enter__(self):
        # Assign the null pointers to stdout
        os.dup2(self.null_fds,1)
        os.close(self.null_fds)

    def __exit__(self, *_):
        # Re-assign the real stdout back
        os.dup2(self.save_fds,1)
        # Close the null file
        os.close(self.save_fds)

def rotate_about_median(xyz, theta):
    """                                                                                                                                             
    rotates xyz by theta around the median along the x, y dimensions                                                                                
    """
    median = np.median(xyz, axis=0)
    centered_xyz = xyz - median
    r_mat = np.eye(3)
    r_mat[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rotated_xyz = centered_xyz.dot(r_mat)
    new_xyz = rotated_xyz + median    
    return new_xyz

def sample_rope_state(demofile, perturb_points=5, min_rad=0, max_rad=.15, rotation=False):
    """
    samples a rope state, by picking a random segment, perturbing, rotating about the median, 
    then setting a random translation such that the rope is essentially within grasp room
    """

    # TODO: pick a random rope initialization
    new_xyz, source_name = load_random_start_segment(demofile)
    perturb_radius = random.uniform(min_rad, max_rad)
    rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                    perturb_peak_dist=perturb_radius,
                                                                    num_perturb_points=perturb_points)
    if rotation:
        rand_theta = rotation*np.random.rand()
        rope_nodes = rotate_about_median(rope_nodes, rand_theta)
    return rope_nodes, source_name
