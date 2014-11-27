from __future__ import division
import cloudprocpy
from rapprentice import berkeley_pr2, cloud_proc_func, PR2, pr2_trajectories

class RobotWorld(object):
    def __init__(self):
    	#move this
        self.pr2 = PR2.PR2()
        self.robot = self.pr2.robot
    
    def observe_cloud(self):
        raise NotImplementedError
    
    def open_gripper(self):
        raise NotImplementedError
    
    def close_gripper(self):
        raise NotImplementedError
    
    def execute_trajectory(self):
        raise NotImplementedError

class RealRobotWorld(RobotWorld):
    def __init__(self):
        super(RealRobotWorld, self).__init__()    
    
    def observe_cloud(self):
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()
        rgb, depth = grabber.getRGBD()
        T_w_k = berkeley_pr2.get_kinect_transform(self.robot)
        full_cloud = cloud_proc_func(rgb, depth, T_w_k)
        return full_cloud
    
    def open_gripper(self, lr):
        gripper = {"l":self.pr2.lgrip, "r":self.pr2.rgrip}[lr]
        gripper.open()
        self.pr2.join_all()
    
    def close_gripper(self, lr):
        gripper = {"l":self.pr2.lgrip, "r":self.pr2.rgrip}[lr]
        gripper.close()
        self.pr2.join_all()
    
    def execute_trajectory(self, full_traj, step_viewer=1, interactive=False,
                           max_cart_vel_trans_traj=.05, sim_callback=None):
    	#resampling
        bodypart2traj = {}
        for lr in "lr":
            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = full_traj.lr2arm_traj[lr]
        pr2_trajectories.follow_body_traj(self.pr2, bodypart2traj)


class ForceRealRobotWorld(RealRobotWorld):
    def __init__(self):
        super(ForceRealRobotWorld, self).__init__()  
    
    def execute_trajectory(self, traj, F, kpl, kvl, gripper, ext, length_total, initial_traj_length, args):
        pgains = np.asarray([2400.0, 1200.0, 1000.0, 700.0, 300.0, 300.0, 300.0])
        dgains = np.asarray([18.0, 10.0, 6.0, 4.0, 6.0, 4.0, 4.0])
        pgainsdiag = np.diag(np.asarray([-2400.0, -1200.0, -1000.0, -700.0, -300.0, -300.0, -300.0]))
        dgainsdiag = np.diag(np.asarray([-18.0, -10.0, -6.0, -4.0, -6.0, -4.0, -4.0]))
        m = np.array([3.33, 1.16, 0.1, 0.25, 0.133, 0.0727, 0.0727]) # masses in joint space (feed forward)
        A = np.diag(m)
        A_inv = np.linalg.inv(A)
        costs = {}
        JKpJ = {}
        JKvJ = {}
        JKfJ = {}
        fake_cost = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        Kp = {}
        Kv = {}
        Kf = {}
        Fe = {}
        for lr in 'lr':
            Kp[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            Kv[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            Kf[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            Fe[lr] = np.asarray([np.zeros((6, 1)) for i in range(length_total)])    

        if args.fitgains:
            redprint("fitting gains")
            Kp['l'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kv['l'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kp['r'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kv['r'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            for lr in 'lr':
                JKfJ[lr] = [np.zeros((7, 7)) for i in range(length_total)]
                JKpJ[lr] = [pgainsdiag for i in range(initial_traj_length)]
                JKvJ[lr] = [dgainsdiag for i in range(initial_traj_length)]
            #import IPython
            #IPython.embed()
            for i in range(length_part):
                Kp['l'].append(np.clip(np.diag(kpl[i]), -np.inf, 0))
                Kv['l'].append(np.clip(np.diag(kvl[i]), -np.inf, 0))
                Kp['r'].append(np.clip(np.diag(kpr[i]), -np.inf, 0))
                Kv['r'].append(np.clip(np.diag(kvr[i]), -np.inf, 0))
                for lr in 'lr':
                    pmin = np.max(np.diag(Kp[lr][initial_traj_length+i])[0:3]) / -100.0
                    pfactor = min(max(pmin, 0.0), 1.0)
                    vmin = np.max(np.diag(Kv[lr][initial_traj_length+i])[0:3]) / -100.0
                    vfactor = min(max(vmin, 0.0), 1.0)
                    #print pfactor
                    #print vfactor
                    JKpJ[lr].append(pgainsdiag*pfactor)
                    JKvJ[lr].append(dgainsdiag*vfactor)
            Kp['l'] = np.asarray(Kp['l'])
            Kv['l'] = np.asarray(Kv['l'])
            Kp['r'] = np.asarray(Kp['r'])
            Kv['r'] = np.asarray(Kv['r'])
            for lr in 'lr':
                JKpJ[lr] = np.asarray(JKpJ[lr])
                JKvJ[lr] = np.asarray(JKvJ[lr])
                JKfJ[lr] = np.asarray(JKfJ[lr])


        if args.fitgainsforce:
            redprint("fitting gains")
            Kp['l'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kv['l'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kf['l'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kp['r'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kv['r'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            Kf['r'] = [np.zeros((6, 6)) for i in range(initial_traj_length)]
            for lr in 'lr':
                JKfJ[lr] = [np.zeros((7, 7)) for i in range(length_total)]
                JKpJ[lr] = [pgainsdiag for i in range(initial_traj_length)]
                JKvJ[lr] = [dgainsdiag for i in range(initial_traj_length)]
            #import IPython
            #IPython.embed()
            for i in range(length_part):
                Kp['l'].append(np.clip(np.diag(kpl[i]), -np.inf, 0))
                Kv['l'].append(np.clip(np.diag(kvl[i]), -np.inf, 0))
                Kf['l'].append(np.clip(np.diag(kfl[i]), -np.inf, 0))
                Kp['r'].append(np.clip(np.diag(kpr[i]), -np.inf, 0))
                Kv['r'].append(np.clip(np.diag(kvr[i]), -np.inf, 0))
                Kf['r'].append(np.clip(np.diag(kfr[i]), -np.inf, 0))
                for lr in 'lr':
                    pmin = np.max(np.diag(Kp[lr][initial_traj_length+i])[0:3]) / -100.0
                    pfactor = min(max(pmin, 0.0), 1.0)
                    vmin = np.max(np.diag(Kv[lr][initial_traj_length+i])[0:3]) / -100.0
                    vfactor = min(max(vmin, 0.0), 1.0)
                    #print pfactor
                    #print vfactor
                    JKpJ[lr].append(pgainsdiag*pfactor)
                    JKvJ[lr].append(dgainsdiag*vfactor)
            Kp['l'] = np.asarray(Kp['l'])
            Kv['l'] = np.asarray(Kv['l'])
            Kf['l'] = np.asarray(Kf['l'])
            Kp['r'] = np.asarray(Kp['r'])
            Kv['r'] = np.asarray(Kv['r'])
            Kf['r'] = np.asarray(Kf['r'])
            for lr in 'lr':
                JKpJ[lr] = np.asarray(JKpJ[lr])
                JKvJ[lr] = np.asarray(JKvJ[lr])
                JKfJ[lr] = np.asarray(JKfJ[lr])

        if args.force:
            for lr in 'lr':
                JKpJ[lr] = np.asarray([pgainsdiag/args.multiplier for i in range(length_total)])
                JKvJ[lr] = np.asarray([dgainsdiag/args.multiplier for i in range(length_total)])
                JKfJ[lr] = np.asarray([np.zeros((7, 7)) for i in range(length_total)])
                Kp[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
                Kv[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
                Kf[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])

        if args.kinematics:
            for lr in 'lr':
                JKpJ[lr] = np.asarray([pgainsdiag for i in range(length_total)])
                JKvJ[lr] = np.asarray([dgainsdiag for i in range(length_total)])
                JKfJ[lr] = np.asarray([np.zeros((7, 7)) for i in range(length_total)])
                Kp[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
                Kv[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
                Kf[lr] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
                F[lr] = np.asarray([np.zeros((6, 1)) for i in range(length_total)])


        if args.onearm:
            traj['l'] = np.asarray([traj['l'][0,:] for i in range(length_total)])
            JKpJ['l'] = np.asarray([pgainsdiag for i in range(length_total)])
            JKvJ['l'] = np.asarray([dgainsdiag for i in range(length_total)])
            JKfJ['l'] = np.asarray([np.zeros((7, 7)) for i in range(length_total)])
            Kp['l'] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            Kv['l'] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            Kf['l'] = np.asarray([np.zeros((6, 6)) for i in range(length_total)])
            F['l'] = np.asarray([np.zeros((6, 1)) for i in range(length_total)])
                
        ### Set all of your gains here and they will be resized/sent to the controller properly
        
        for lr in 'lr':
            JKvJ[lr] = np.resize(JKvJ[lr], (1, 49*length_total))[0]
            JKpJ[lr] = np.resize(JKpJ[lr], (1, 49*length_total))[0]
            JKfJ[lr] = np.resize(JKfJ[lr], (1, 49*length_total))[0]
            traj[lr] = np.resize(traj[lr], (1, traj[lr].shape[0]*7))[0]
            Fe[lr] = np.resize(Fe[lr], (1, Fe[lr].shape[0]*6))[0]
            ext[lr] = np.resize(ext[lr], (1, ext[lr].shape[0]*7))[0]
            F[lr] = np.resize(F[lr], (1, F[lr].shape[0]*6))[0]
            Kp[lr] = np.resize(Kp[lr], (1, 36 * length_total))[0]
            Kf[lr] = np.resize(Kf[lr], (1, 36 * length_total))[0]
            Kv[lr] = np.resize(Kv[lr], (1, 36 * length_total))[0]
        
        gripper = np.resize(gripper, (1, gripper.shape[0]))[0]


        # [traj, Kp, Kv, F, use_force, seconds]
        data = np.zeros((1, length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49+36+1)))

        data[0][0:length_total*7] = traj['r']
        data[0][length_total*7:length_total*(7+49)] = JKpJ['r']
        data[0][length_total*(7+49):length_total*(7+49+49)] = JKvJ['r']
        data[0][length_total*(7+49+49):length_total*(7+49+49+6)] = F['r']
        data[0][length_total*(7+49+49+6):length_total*(7+49+49+6+36)] = Kp['r']
        data[0][length_total*(7+49+49+6+36):length_total*(7+49+49+6+36+36)] = Kv['r']

        data[0][length_total*(7+49+49+6+36+36):length_total*(7+49+49+6+36+36+7)] = ext['r']
        data[0][length_total*(7+49+49+6+36+36+7):length_total*(7+49+49+6+36+36+7+6)] = Fe['r']
        data[0][length_total*(7+49+49+6+36+36+7+6):length_total*(7+49+49+6+36+36+7+6+49)] = JKfJ['r']
        data[0][length_total*(7+49+49+6+36+36+7+6+49):length_total*(7+49+49+6+36+36+7+6+49+36)] = Kf['r']

        data[0][length_total*(7+49+49+6+36+36+7+6+49+36):length_total*(7+49+49+6+36+36+7+6+49+36+7)] = traj['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7):length_total*(7+49+49+6+36+36+7+6+49+36+7+49)] = JKpJ['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49)] = JKvJ['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6)] = F['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36)] = Kp['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36)] = Kv['l']

        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7)] = ext['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6)] = Fe['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49)] = JKfJ['l']
        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49+36)] = Kf['l']

        data[0][length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49+36):length_total*(7+49+49+6+36+36+7+6+49+36+7+49+49+6+36+36+7+6+49+36+1)] = gripper

        msg = Float64MultiArray()
        msg.data = data[0].tolist()
        pub = rospy.Publisher("/controller_data", Float64MultiArray)
        redprint("Press enter to start trajectory")
        if args.onearm:
            time.sleep(5)
            pub.publish(msg)
        else:    
            for lr in 'lr':
                self.open_gripper(lr)
            time.sleep(5)
            pub.publish(msg)
            listener()

