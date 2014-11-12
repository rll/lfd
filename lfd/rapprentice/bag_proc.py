from lfd.rapprentice import ros2rave, func_utils, berkeley_pr2
import fastrapp
import numpy as np
import cv2
import openravepy
import os.path as osp

def extract_joints(bag):
    """returns (names, traj) 
    """
    traj = []
    stamps = []
    for (_, msg, _) in bag.read_messages(topics=['/joint_states']):        
        traj.append(msg.position)
        stamps.append(msg.header.stamp.to_sec())
    assert len(traj) > 0
    names = msg.name
    return names, stamps, traj
    
def extract_joy(bag):
    """sounds morbid    
    """

    stamps = []
    meanings = []
    button2meaning = {
        12: "look",
        0: "start",
        3: "stop",
        7: "l_open",
        5: "l_close",
        15: "r_open",
        13: "r_close",
        14: "done"
    }
    check_buttons = button2meaning.keys()
    message_stream = bag.read_messages(topics=['/joy'])
    (_,last_msg,_) = message_stream.next()
    for (_, msg, _) in message_stream:
        for i in check_buttons:
            if msg.buttons[i] and not last_msg.buttons[i]:
                stamps.append(msg.header.stamp.to_sec())
                meanings.append(button2meaning[i])
        last_msg = msg
        
    return stamps, meanings

        
def find_disjoint_subsequences(li, seq):
    """
    Returns a list of tuples (i,j,k,...) so that seq == (li[i], li[j], li[k],...)
    Greedily find first tuple, then second, etc.
    """
    subseqs = []
    cur_subseq_inds = []
    for (i_el, el) in enumerate(li):
        if el == seq[len(cur_subseq_inds)]:
            cur_subseq_inds.append(i_el)
            if len(cur_subseq_inds) == len(seq):
                subseqs.append(cur_subseq_inds)
                cur_subseq_inds = []
    return subseqs
    
def joy_to_annotations(stamps, meanings):
    """return a list of dicts giving info for each segment
    [{"look": 1234, "start": 2345, "stop": 3456},...]
    """
    out = []
    ind_tuples = find_disjoint_subsequences(meanings, ["look","start","stop"])
    for tup in ind_tuples:
        out.append({"look":stamps[tup[0]], "start":stamps[tup[1]], "stop":stamps[tup[2]]})
    return out

def add_kinematics_to_group(group, linknames, manipnames, jointnames, robot):
    "do forward kinematics on those links"
    if robot is None: robot = get_robot()
    r2r = ros2rave.RosToRave(robot, group["joint_states"]["name"])
    link2hmats = dict([(linkname, []) for linkname in linknames])
    links = [robot.GetLink(linkname) for linkname in linknames]
    rave_traj = []
    rave_inds = r2r.rave_inds
    for ros_vals in group["joint_states"]["position"]:
        r2r.set_values(robot, ros_vals)
        rave_vals = r2r.convert(ros_vals)
        robot.SetDOFValues(rave_vals, rave_inds)
        rave_traj.append(rave_vals)
        for (linkname,link) in zip(linknames, links):
            link2hmats[linkname].append(link.GetTransform())
    for (linkname, hmats) in link2hmats.items():
        group.create_group(linkname)
        group[linkname]["hmat"] = np.array(hmats)      
        
    rave_traj = np.array(rave_traj)
    rave_ind_list = list(rave_inds)
    for manipname in manipnames:
        arm_inds = robot.GetManipulator(manipname).GetArmIndices()
        group[manipname] = rave_traj[:,[rave_ind_list.index(i) for i in arm_inds]]
        
    for jointname in jointnames:
        joint_ind = robot.GetJointIndex(jointname)
        group[jointname] = rave_traj[:,rave_ind_list.index(joint_ind)]
        
    
    
    
@func_utils.once
def get_robot():
    env = openravepy.Environment()
    env.Load("robots/pr2-beta-static.zae")
    robot = env.GetRobots()[0]
    return robot
    
def add_bag_to_hdf(bag, annotations, hdfroot, demo_name):
    joint_names, stamps, traj = extract_joints(bag)
    traj = np.asarray(traj)
    stamps = np.asarray(stamps)
    
    robot = get_robot()

    for seg_info in annotations:


        group = hdfroot.create_group(demo_name + "_" + seg_info["name"])
    
        start = seg_info["start"]
        stop = seg_info["stop"]
        
        [i_start, i_stop] = np.searchsorted(stamps, [start, stop])
        
        stamps_seg = stamps[i_start:i_stop+1]
        traj_seg = traj[i_start:i_stop+1]
        sample_inds = fastrapp.resample(traj_seg, np.arange(len(traj_seg)), .01, np.inf, np.inf)
        print "trajectory has length", len(sample_inds),len(traj_seg)

    
        traj_ds = traj_seg[sample_inds,:]
        stamps_ds = stamps_seg[sample_inds]
    
        group["description"] = seg_info["description"]
        group["stamps"] = stamps_ds
        group.create_group("joint_states")
        group["joint_states"]["name"] = joint_names
        group["joint_states"]["position"] = traj_ds
        link_names = ["l_gripper_tool_frame","r_gripper_tool_frame","l_gripper_r_finger_tip_link","l_gripper_l_finger_tip_frame","r_gripper_r_finger_tip_link","r_gripper_l_finger_tip_frame"]
        special_joint_names = ["l_gripper_joint", "r_gripper_joint"]
        manip_names = ["leftarm", "rightarm"]
        
        add_kinematics_to_group(group, link_names, manip_names, special_joint_names, robot)

def get_video_frames(video_dir, frame_stamps):
    video_stamps = np.loadtxt(osp.join(video_dir,"stamps.txt"))
    frame_inds = np.searchsorted(video_stamps, frame_stamps)
    
    rgbs = []
    depths = []
    for frame_ind in frame_inds:
        rgb = cv2.imread(osp.join(video_dir,"rgb%i.jpg"%frame_ind))
        assert rgb is not None
        rgbs.append(rgb)
        depth = cv2.imread(osp.join(video_dir,"depth%i.png"%frame_ind),2)
        assert depth is not None
        depths.append(depth)
    return rgbs, depths


def add_rgbd_to_hdf(video_dir, annotations, hdfroot, demo_name):
    
    frame_stamps = [seg_info["look"] for seg_info in annotations]
    
    rgb_imgs, depth_imgs = get_video_frames(video_dir, frame_stamps)
    
    for (i_seg, seg_info) in enumerate(annotations):        
        group = hdfroot[demo_name + "_" + seg_info["name"]]
        group["rgb"] = rgb_imgs[i_seg]
        group["depth"] = depth_imgs[i_seg]
        robot = get_robot()
        r2r = ros2rave.RosToRave(robot, group["joint_states"]["name"])
        r2r.set_values(robot, group["joint_states"]["position"][0])
        T_w_h = robot.GetLink("head_plate_frame").GetTransform()
        T_w_k = T_w_h.dot(berkeley_pr2.T_h_k)
        group["T_w_k"] = T_w_k

