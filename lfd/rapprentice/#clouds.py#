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
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)
    cloud = cloudprocpy.downsampleCloud(cloud, v)
    return cloud.to2dArray()[:,:3]

    