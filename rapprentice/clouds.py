import numpy as np

cx = 320.-.5
cy = 240.-.5
DEFAULT_F = 535.

def xyZ_to_XY(x,y,Z,f=DEFAULT_F):
    X = (x - cx)*(Z/f)
    Y = (y - cy)*(Z/f)
    return (X,Y)

def XYZ_to_xy(X,Y,Z,f=DEFAULT_F):
    x = X*(f/Z) + cx
    y = Y*(f/Z) + cy
    return (x,y)

def depth_to_xyz(depth,f=DEFAULT_F):
    x,y = np.meshgrid(np.arange(640), np.arange(480))
    assert depth.shape == (480, 640)
    XYZ = np.empty((480,640,3))
    Z = XYZ[:,:,2] = depth / 1000. # convert mm -> meters
    XYZ[:,:,0] = (x - cx)*(Z/f)
    XYZ[:,:,1] = (y - cy)*(Z/f)

    return XYZ
    
def downsample(xyz, v):
    import cloudprocpy
    if xyz.shape[1] == 3:
        cloud = cloudprocpy.CloudXYZ()
        xyz1 = np.ones((len(xyz),4),'float')
        xyz1[:,:3] = xyz
        cloud.from2dArray(xyz1)
        cloud = cloudprocpy.downsampleCloud(cloud, v)
        return cloud.to2dArray()[:,:3]
    else:
        # rgb fields needs to be packed and upacked as described in here
        # http://docs.pointclouds.org/1.7.0/structpcl_1_1_point_x_y_z_r_g_b.html
        xyzrgb = xyz
        n = xyzrgb.shape[0]
        cloud = cloudprocpy.CloudXYZRGB()
        xyzrgb1 = np.ones((n,8),'float')
        xyzrgb1[:,:3] = xyzrgb[:,:3]
        xyzrgb1[:,4] = cloudprocpy.packRGBs(xyzrgb[:,3:] * 255.0)
        xyzrgb1[:,5:] = np.zeros((n,3)) # padding that is not used. set to zero just in case
        cloud.from2dArray(xyzrgb1)
        cloud = cloudprocpy.downsampleColorCloud(cloud, v)
        xyzrgb1 = cloud.to2dArray()
        return np.c_[xyzrgb1[:,:3], cloudprocpy.unpackRGBs(xyzrgb1[:,4]) / 255.0]
    