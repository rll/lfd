import cloudprocpy
from lfd.rapprentice import berkeley_pr2, clouds
import cv2, numpy as np
import skimage.morphology as skim
DEBUG_PLOTS = False


def extract_red(rgb, depth, T_w_k):
    """
    extract red points and downsample
    """
        
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    
    valid_mask = depth > 0
    
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    z = xyz_w[:,:,2]   
    z0 = xyz_k[:,:,2]
    if DEBUG_PLOTS:
        cv2.imshow("z0",z0/z0.max())
        cv2.imshow("z",z/z.max())
        cv2.imshow("rgb", rgb)
        cv2.waitKey()
    
    height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter
    
    
    good_mask = red_mask & height_mask & valid_mask
    if DEBUG_PLOTS:
        cv2.imshow("red",red_mask.astype('uint8')*255)
        cv2.imshow("above_table", height_mask.astype('uint8')*255)
        cv2.imshow("red and above table", good_mask.astype('uint8')*255)
        print "press enter to continue"
        cv2.waitKey()
        
        
    

    good_xyz = xyz_w[good_mask]
    

    return clouds.downsample(good_xyz, .01)
    
    
def grabcut(rgb, depth, T_w_k):
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

    valid_mask = depth > 0

    import interactive_roi as ir
    xys = ir.get_polyline(rgb, "rgb")
    xy_corner1 = np.clip(np.array(xys).min(axis=0), [0,0], [639,479])
    xy_corner2 = np.clip(np.array(xys).max(axis=0), [0,0], [639,479])
    polymask = ir.mask_from_poly(xys)
    #cv2.imshow("mask",mask)
        
    xy_tl = np.array([xy_corner1, xy_corner2]).min(axis=0)
    xy_br = np.array([xy_corner1, xy_corner2]).max(axis=0)

    xl, yl = xy_tl
    w, h = xy_br - xy_tl
    mask = np.zeros((h,w),dtype='uint8')    
    mask[polymask[yl:yl+h, xl:xl+w] > 0] = cv2.GC_PR_FGD
    print mask.shape
    #mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD

    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))    
    cv2.grabCut(rgb[yl:yl+h, xl:xl+w, :],mask,(0,0,0,0),tmp1, tmp2,10,mode=cv2.GC_INIT_WITH_MASK)

    mask = mask % 2
    #mask = ndi.binary_erosion(mask, utils_images.disk(args.erode)).astype('uint8')
    contours = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(rgb[yl:yl+h, xl:xl+w, :],contours,-1,(0,255,0),thickness=2)
    
    cv2.imshow('rgb', rgb)
    print "press enter to continue"
    cv2.waitKey()

    zsel = xyz_w[yl:yl+h, xl:xl+w, 2]
    mask = (mask%2==1) & np.isfinite(zsel)# & (zsel - table_height > -1)
    mask &= valid_mask[yl:yl+h, xl:xl+w]
    
    xyz_sel = xyz_w[yl:yl+h, xl:xl+w,:][mask.astype('bool')]
    return clouds.downsample(xyz_sel, .01)
    #rgb_sel = rgb[yl:yl+h, xl:xl+w,:][mask.astype('bool')]
        
def filter_green(rgb, depth, T_w_k):  
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2] 
    h_mask = (h > 50) & (h < 110)   
    s_mask = (s > 40) & (h < 160) 
    v_mask = (v > 40)  & (v < 240)  
    # IPython.embed()
    x = 100
    y = 240   
    xwin = [x,x+50]   
    ywin = [y,y+50] 
    rgb = rgb[ywin[0]:ywin[1],xwin[0]:xwin[1]] 
    h = h[ywin[0]:ywin[1],xwin[0]:xwin[1]]
    s = s[ywin[0]:ywin[1],xwin[0]:xwin[1]]
    v = v[ywin[0]:ywin[1],xwin[0]:xwin[1]]  
    # IPython.embed()
    green_mask = h_mask & s_mask & v_mask
    green_filter = np.invert(green_mask)
    green_filter[350:480,500:640] = 0 
    valid_mask = depth > 0
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]  
    z = xyz_w[:,:,2] 
    z0 = xyz_k[:,:,2]
    #height_mask = xyz_w[:,:,2] > .7 # TODO pass in parameter 
    #height_mask_bottom = xyz_w[:,:,2] < 0.8 
    x_mask = xyz_w[:,:,0] < 1
    x_mask_1 = xyz_w[:,:,0] > 0.3
    y_mask_2 = xyz_w[:,:,1] > -0.6
    z_mask = xyz_w[:,:,2] < 0.85
    # IPython.embed()
    good_mask = green_filter & valid_mask &  x_mask & x_mask_1  #& y_mask_2 #& z_mask
    # good_mask = valid_mask
    good_mask =   skim.remove_small_objects(good_mask,min_size=84)  
     
    cv2.imshow("final",good_mask.astype('uint8')*255)
    cv2.imshow("rgb", rgb)
    cv2.waitKey()
    good_xyz = xyz_w[good_mask] 
    # good_rgb = rgb[good_mask]
    # good_xyzrgb = np.c_[good_xyz, good_rgb]
    return good_xyz


def extract_red_alphashape(cloud, robot):
    """
    extract red, get alpha shape, downsample
    """
    raise NotImplementedError
    
    # downsample cloud
    cloud_ds = cloudprocpy.downsampleCloud(cloud, .01)
    
    # transform into body frame
    xyz1_kinect = cloud_ds.to2dArray()
    xyz1_kinect[:,3] = 1
    T_w_k = berkeley_pr2.get_kinect_transform(robot)
    xyz1_robot = xyz1_kinect.dot(T_w_k.T)
    
    # compute 2D alpha shape
    xyz1_robot_flat = xyz1_robot.copy()
    xyz1_robot_flat[:,2] = 0 # set z coordinates to zero
    xyz1_robot_flatalphashape = cloudprocpy.computeAlphaShape(xyz1_robot_flat)
    
    # unfortunately pcl alpha shape func throws out the indices, so we have to use nearest neighbor search
    cloud_robot_flatalphashape = cloudprocpy.CloudXYZ()
    cloud_robot_flatalphashape.from2dArray(xyz1_robot_flatalphashape)
    cloud_robot_flat = cloudprocpy.CloudXYZ()
    cloud_robot_flat.from2dArray(xyz1_robot_flat)
    alpha_inds = cloudprocpy.getNearestNeighborIndices(xyz1_robot_flatalphashape, xyz1_robot_flat)

    xyz_robot_alphashape = xyz1_robot_flatalphashape[:,:3]
    
    # put back z coordinate
    xyz_robot_alphashape[:,2] = xyz1_robot[alpha_inds,2] 

    return xyz_robot_alphashape[:,:3]
