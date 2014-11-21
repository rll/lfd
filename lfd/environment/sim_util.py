# Contains useful functions for PR2 rope tying simulation
# The purpose of this class is to eventually consolidate
# the various instantiations of do_task_eval.py

import bulletsimpy
import openravepy
import numpy as np
from numpy import asarray
import re
import random

from lfd.rapprentice import animate_traj, ropesim, math_utils as mu, plotting_openrave
from lfd.util import util
import settings
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
        self.radius       = settings.ROPE_RADIUS
        self.angStiffness = settings.ROPE_ANG_STIFFNESS
        self.angDamping   = settings.ROPE_ANG_DAMPING
        self.linDamping   = settings.ROPE_LIN_DAMPING
        self.angLimit     = settings.ROPE_ANG_LIMIT
        self.linStopErp   = settings.ROPE_LIN_STOP_ERP
        self.mass         = settings.ROPE_MASS

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
    l_openings = np.flatnonzero((lgrip[1:] >= settings.GRIPPER_OPEN_CLOSE_THRESH) & (lgrip[:-1] < settings.GRIPPER_OPEN_CLOSE_THRESH))
    r_openings = np.flatnonzero((rgrip[1:] >= settings.GRIPPER_OPEN_CLOSE_THRESH) & (rgrip[:-1] < settings.GRIPPER_OPEN_CLOSE_THRESH))
    l_closings = np.flatnonzero((lgrip[1:] < settings.GRIPPER_OPEN_CLOSE_THRESH) & (lgrip[:-1] >= settings.GRIPPER_OPEN_CLOSE_THRESH))
    r_closings = np.flatnonzero((rgrip[1:] < settings.GRIPPER_OPEN_CLOSE_THRESH) & (rgrip[:-1] >= settings.GRIPPER_OPEN_CLOSE_THRESH))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def split_trajectory_by_lr_gripper(seg_info, lr):
    grip = asarray(seg_info["%s_gripper_joint"%lr])

    n_steps = len(grip)

    # indices BEFORE transition occurs
    openings = np.flatnonzero((grip[1:] >= settings.GRIPPER_OPEN_CLOSE_THRESH) & (grip[:-1] < settings.GRIPPER_OPEN_CLOSE_THRESH))
    closings = np.flatnonzero((grip[1:] < settings.GRIPPER_OPEN_CLOSE_THRESH) & (grip[:-1] >= settings.GRIPPER_OPEN_CLOSE_THRESH))

    before_transitions = np.r_[openings, closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def get_opening_closing_inds(finger_traj):
    
    mult = 5.0
    GRIPPER_L_FINGER_OPEN_CLOSE_THRESH = mult * settings.GRIPPER_OPEN_CLOSE_THRESH

    # indices BEFORE transition occurs
    opening_inds = np.flatnonzero((finger_traj[1:] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    closing_inds = np.flatnonzero((finger_traj[1:] < GRIPPER_L_FINGER_OPEN_CLOSE_THRESH) & (finger_traj[:-1] >= GRIPPER_L_FINGER_OPEN_CLOSE_THRESH))
    
    return opening_inds, closing_inds

def gripper_joint2gripper_l_finger_joint_values(gripper_joint_vals):
    """
    Only the %s_gripper_l_finger_joint%lr can be controlled (this is the joint returned by robot.GetManipulator({"l":"leftarm", "r":"rightarm"}[lr]).GetGripperIndices())
    The rest of the gripper joints (like %s_gripper_joint%lr) are mimiced and cannot be controlled directly
    """
    mult = 5.0
    gripper_l_finger_joint_vals = mult * gripper_joint_vals
    return gripper_l_finger_joint_vals

def binarize_gripper(angle):
    return angle > settings.GRIPPER_OPEN_CLOSE_THRESH

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
        sim_env.step()
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
        sim_env.step()

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
        util.yellowprint("Some steps of the trajectory were not executed because the gripper was pulling the rope too tight.")
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
    if sim_env.constraints['l'] and sim_env.constraints['r']:
        ee_trajs = {}
        for lr in 'lr':
            ee_trajs[lr] = get_ee_traj(sim_env.robot, lr, full_traj, ee_link_name_fmt="%s_gripper_l_finger_tip_link")
        min_length = np.inf
        raise NameError('Hello')
        hs = sim_env.sim.rope.GetHalfHeights()
        rope_links = sim_env.sim.rope.GetKinBody().GetLinks()
        for l_rope_link in sim_env.constraints_links['l']:
            if l_rope_link not in rope_links:
                continue
            for r_rope_link in sim_env.constraints_links['r']:
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
    if sim_env.constraints['l'] and sim_env.constraints['r']:
        ee_trajs = {}
        for lr in 'lr':
            ee_trajs[lr] = get_ee_traj(sim_env, lr, full_traj, ee_link_name_fmt="%s_gripper_l_finger_tip_link")
        min_length = np.inf
        hs = sim_env.sim.rope.GetHalfHeights()
        rope_links = sim_env.sim.rope.GetKinBody().GetLinks()
        for l_rope_link in sim_env.constraints_links['l']:
            if l_rope_link not in rope_links:
                continue
            for r_rope_link in sim_env.constraints_links['r']:
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
            util.yellowprint("The grippers of the trajectory goes too far apart. Some steps of the trajectory are being removed.")
    return full_traj

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz'][:,:3]

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
    rope_params.radius       = settings.ROPE_RADIUS
    rope_params.angStiffness = settings.ROPE_ANG_STIFFNESS
    rope_params.angDamping   = settings.ROPE_ANG_DAMPING
    rope_params.linDamping   = settings.ROPE_LIN_DAMPING
    rope_params.angLimit     = settings.ROPE_ANG_LIMIT
    rope_params.linStopErp   = settings.ROPE_LIN_STOP_ERP
    rope_params.mass         = settings.ROPE_MASS

    if params_id == 'default':
        pass
    elif params_id == 'thick':
        rope_params.radius = settings.ROPE_RADIUS_THICK
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

def draw_axis(sim_env, hmat, arrow_length=.1, arrow_width=0.005):
    handles = []
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,0]*arrow_length, arrow_width, (1,0,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,1]*arrow_length, arrow_width, (0,1,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,2]*arrow_length, arrow_width, (0,0,1,1)))
    return handles

def draw_finger_pts_traj(sim_env, flr2finger_pts_traj, color):
    handles = []
    for finger_lr, pts_traj in flr2finger_pts_traj.items():
        for pts in pts_traj:
            handles.append(sim_env.env.drawlinestrip(np.r_[pts, pts[0][None,:]], 1, color))
    return handles

