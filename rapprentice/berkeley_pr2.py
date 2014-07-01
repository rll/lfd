"""
put Berkeley-specific parameters here
"""



import numpy as np



"""
Berkeley PR2's kinect transform
Here's what's in the URDF:
    <joint name="camera_rgb_optical_frame_joint" type="fixed">
    <parent link="head_plate_frame"/>
    <child link="camera_link"/>
    <origin rpy="0.005 0.035 0.02" xyz="-0.19 0.045 0.22"/>
    </joint>
    <link name="camera_link" type="camera"/>
"""

#T_b_o = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
#T_b_o = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])

T_h_k = np.array([[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
 [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
 [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
 [ 0.,          0.,          0.,          1.        ]])

f = 544.260779961

def get_kinect_transform(robot):    
    T_w_h = robot.GetLink("head_plate_frame").GetTransform()    
    T_w_k = T_w_h.dot(T_h_k)
    return T_w_k

       