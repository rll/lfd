"""
Plotting functions using matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import art3d
import colorsys

def plot_warped_grid_2d(f, mins, maxes, grid_res=None, nfine=30, color='gray', flipax=False, draw=True):
    xmin, ymin = mins
    xmax, ymax = maxes
    ncoarse = 10

    if grid_res is None:
        xcoarse = np.linspace(xmin, xmax, ncoarse)
        ycoarse = np.linspace(ymin, ymax, ncoarse)
    else:
        xcoarse = np.arange(xmin, xmax, grid_res)
        ycoarse = np.arange(ymin, ymax, grid_res)
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)

    lines = []

    sgn = -1 if flipax else 1

    for x in xcoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = x
        xy[:,1] = yfine
        lines.append(f(xy)[:,::sgn])

    for y in ycoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = xfine
        xy[:,1] = y
        lines.append(f(xy)[:,::sgn])        

    lc = matplotlib.collections.LineCollection(lines,colors=color,lw=1)
    ax = plt.gca()
    ax.add_collection(lc)
    if draw:
        plt.draw()

# almost copied from plotting_openrave
def plot_warped_grid_3d(f, mins, maxes, xres = .1, yres = .1, zres = .04, color = 'gray', draw=True):
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    nfine = 30
    xcoarse = np.arange(xmin, xmax, xres)
    xmax = xcoarse[-1];
    ycoarse = np.arange(ymin, ymax, yres)
    ymax = ycoarse[-1];
    if zres == -1:
        zcoarse = [(zmin+zmax)/2.]
    else:
        zcoarse = np.arange(zmin, zmax, zres)
        zmax = zcoarse[-1];
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    lc = art3d.Line3DCollection(lines,colors=color,lw=1)
    ax = plt.gca()
    ax.add_collection(lc)
    if draw:
        plt.draw()


def plot_warped_grid_proj_2d(f, mins, maxes, z=.0, xres = .1, yres = .1, color = 'gray', draw=True):
    xmin, ymin = mins
    xmax, ymax = maxes

    nfine = 30
    xcoarse = np.arange(xmin, xmax, xres)
    xmax = xcoarse[-1];
    ycoarse = np.arange(ymin, ymax, yres)
    ymax = ycoarse[-1];
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    
    lines = []
    for y in ycoarse:
        xyz = np.zeros((nfine, 3))
        xyz[:,0] = xfine
        xyz[:,1] = y
        xyz[:,2] = z
        lines.append(f(xyz)[:,:2])
        
    for x in xcoarse:
        xyz = np.zeros((nfine, 3))
        xyz[:,0] = x
        xyz[:,1] = yfine
        xyz[:,2] = z
        lines.append(f(xyz)[:,:2])

    lc = matplotlib.collections.LineCollection(lines,colors=color,lw=1)
    ax = plt.gca()
    ax.add_collection(lc)
    if draw:
        plt.draw()

def plot_tps_registration(x_nd, y_md, f, res = (.1, .1, .04), x_color=None, y_color=None, proj_2d=False):
    """
    Plots warp visualization
    x_nd: source points plotted with ',' and x_color (or red if not especified)
    y_md: target points plotted with '+' and y_color (or blue if not especified)
    warped points plotted with 'o' and x_color (or green if not especified)
    proj_2d: if points are in R^3 and proj_2d is True, the plot is projected to the xy-plane
    """
    _,d = x_nd.shape
    
    if x_color == None:
        x_color = (1,0,0,1)
        xwarped_color = (0,1,0,1)
    else:
        xwarped_color = x_color
    if y_color == None:
        y_color = (0,0,1,1)
    
    if d == 3:
        if proj_2d:
            plot_tps_registration_proj_2d(x_nd, y_md, f, res, x_color, y_color, xwarped_color)
        else:
            plot_tps_registration_3d(x_nd, y_md, f, res, x_color, y_color, xwarped_color)
    else:
        plot_tps_registration_2d(x_nd, y_md, f, x_color, y_color, xwarped_color)

def plot_tps_registration_2d(x_nd, y_md, f, x_color, y_color, xwarped_color):
    # set interactive
    plt.ion()
    
    fig = plt.figure('2d plot')
    fig.clear()

    plt.subplot(121, aspect='equal')
    plt.scatter(x_nd[:,0], x_nd[:,1], c=x_color, edgecolors=x_color, marker=',', s=5)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    plot_warped_grid_2d(lambda xyz: xyz, grid_mins, grid_maxs, draw=False)
    
    plt.subplot(122, aspect='equal')
    plt.scatter(y_md[:,0], y_md[:,1], c=y_color, marker='+', s=50)
    xwarped_nd = f.transform_points(x_nd)
    plt.scatter(xwarped_nd[:,0], xwarped_nd[:,1], edgecolors=xwarped_color, facecolors='none', marker='o', s=50)
    
    plot_warped_grid_2d(f.transform_points, grid_mins, grid_maxs, draw=False)
    
    plt.draw()

def plot_tps_registration_3d(x_nd, y_md, f, res, x_color, y_color, xwarped_color):
    # set interactive
    plt.ion()
     
    fig = plt.figure('3d plot')
    fig.clear()

    ax = fig.add_subplot(121, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(x_nd[:,0], x_nd[:,1], x_nd[:,2], c=x_color, edgecolors=x_color, marker=',', s=5)

    # manually set axes limits at a cube's bounding box since matplotlib doesn't correctly set equal axis in 3D
    xwarped_nd = f.transform_points(x_nd)
    max_pts = np.r_[x_nd, y_md, xwarped_nd].max(axis=0)
    min_pts = np.r_[x_nd, y_md, xwarped_nd].min(axis=0)
    max_range = (max_pts - min_pts).max()
    center = 0.5*(max_pts + min_pts)
    ax.set_xlim(center[0] - 0.5*max_range, center[0] + 0.5*max_range)
    ax.set_ylim(center[1] - 0.5*max_range, center[1] + 0.5*max_range)
    ax.set_zlim(center[2] - 0.5*max_range, center[2] + 0.5*max_range)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    plot_warped_grid_3d(lambda xyz: xyz, grid_mins, grid_maxs, xres=res[0], yres=res[1], zres=res[2], draw=False)

    ax = fig.add_subplot(122, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(y_md[:,0], y_md[:,1], y_md[:,2], c=y_color, marker='+', s=50)
    xwarped_nd = f.transform_points(x_nd)
    ax.scatter(xwarped_nd[:,0], xwarped_nd[:,1], xwarped_nd[:,2], edgecolors=xwarped_color, facecolors='none', marker='o', s=50)

    ax.set_xlim(center[0] - 0.5*max_range, center[0] + 0.5*max_range)
    ax.set_ylim(center[1] - 0.5*max_range, center[1] + 0.5*max_range)
    ax.set_zlim(center[2] - 0.5*max_range, center[2] + 0.5*max_range)

    plot_warped_grid_3d(f.transform_points, grid_mins, grid_maxs, xres=res[0], yres=res[1], zres=res[2], draw=False)
    
    plt.draw()

def plot_tps_registration_proj_2d(x_nd, y_md, f, res, x_color, y_color, xwarped_color):
    # set interactive
    plt.ion()
    
    fig = plt.figure('2d projection plot')
    fig.clear()
    
    plt.subplot(121, aspect='equal')
    plt.scatter(x_nd[:,0], x_nd[:,1], c=x_color, edgecolors=x_color, marker=',', s=5)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    x_median = np.median(x_nd, axis=0)
    plot_warped_grid_proj_2d(lambda xyz: xyz, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    
    plt.subplot(122, aspect='equal')
    plt.scatter(y_md[:,0], y_md[:,1], c=y_color, marker='+', s=50)
    xwarped_nd = f.transform_points(x_nd)
    plt.scatter(xwarped_nd[:,0], xwarped_nd[:,1], edgecolors=xwarped_color, facecolors='none', marker='o', s=50)

    plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    
    plt.draw()
    
def plot_tps_registration_segment_proj_2d(rope_nodes0, rope_nodes1, cloud0, cloud1, corr_nm, corr_nm_aug, f, pts_segmentation_inds0, pts_segmentation_inds1):
    # set interactive
    plt.ion()
    
    fig = plt.figure('rope nodes segmentations')
    fig.clear()
    
    plt.subplot(221, aspect='equal')
    if cloud0 is not None:
        plt.scatter(cloud0[:,0], cloud0[:,1], c=(1,0,0), edgecolors=(1,0,0), marker=',', s=1)
#     for i, (i_start, i_end) in enumerate(zip(pts_segmentation_inds0[:-1], pts_segmentation_inds0[1:])):
#         color = 'r' if len(pts_segmentation_inds0)<=2 else np.tile(np.array(colorsys.hsv_to_rgb(float(i)/(len(pts_segmentation_inds0)-2),1,1)), (i_end-i_start,1))
#         plt.scatter(rope_nodes0[i_start:i_end,0], rope_nodes0[i_start:i_end,1], c=color, edgecolors=color, marker=',', s=1)
    grid_means = .5 * (rope_nodes0.max(axis=0) + rope_nodes0.min(axis=0))
    grid_mins = grid_means - (rope_nodes0.max(axis=0) - rope_nodes0.min(axis=0))
    grid_maxs = grid_means + (rope_nodes0.max(axis=0) - rope_nodes0.min(axis=0))
    plot_warped_grid_proj_2d(lambda xyz: xyz, grid_mins[:2], grid_maxs[:2], z=np.median(rope_nodes0, axis=0)[2], xres=.05, yres=.05, draw=False)
    
    plt.subplot(222, aspect='equal')
    if cloud1 is not None:
        plt.scatter(cloud1[:,0], cloud1[:,1], c=(0,0,1), edgecolors=(0,0,1), marker=',', s=1)
    if f is not None and cloud0 is not None:
        warped_cloud0 = f.transform_points(cloud0)
        plt.scatter(warped_cloud0[:,0], warped_cloud0[:,1], c=(0,1,0), edgecolors=(0,1,0), marker=',', s=1)
#     for i, (i_start, i_end) in enumerate(zip(pts_segmentation_inds1[:-1], pts_segmentation_inds1[1:])):
#         color = 'r' if len(pts_segmentation_inds1)<=2 else np.tile(np.array(colorsys.hsv_to_rgb(float(i)/(len(pts_segmentation_inds1)-2),1,1)), (i_end-i_start,1))
#         plt.scatter(rope_nodes1[i_start:i_end,0], rope_nodes1[i_start:i_end,1], c=color, edgecolors=color, marker=',', s=1)
    
    if corr_nm is not None or corr_nm_aug is not None:
        plt.subplot(223, aspect='equal')
        if corr_nm_aug is not None:
            cloud1_resampled = corr_nm_aug.dot(cloud1)
            plt.scatter(cloud1_resampled[:,0], cloud1_resampled[:,1], c=(0,0,1), edgecolors=(0,0,1), marker=',', s=1)
        if f is not None and cloud0 is not None:
            warped_cloud0 = f.transform_points(cloud0)
            plt.scatter(warped_cloud0[:,0], warped_cloud0[:,1], c=(0,1,0), edgecolors=(0,1,0), marker=',', s=1)
#         if corr_nm is not None:
#             rope_nodes1_resampled = corr_nm.dot(rope_nodes1)
#             for i, (i_start, i_end) in enumerate(zip(pts_segmentation_inds0[:-1], pts_segmentation_inds0[1:])):
#                 color = 'r' if len(pts_segmentation_inds0)<=2 else np.tile(np.array(colorsys.hsv_to_rgb(float(i)/(len(pts_segmentation_inds0)-2),1,1)), (i_end-i_start,1))
#                 plt.scatter(rope_nodes1_resampled[i_start:i_end,0], rope_nodes1_resampled[i_start:i_end,1], c=color, edgecolors=color, marker=',', s=1)
    
    if f is not None:
        plt.subplot(224, aspect='equal')
        warped_cloud0 = f.transform_points(cloud0)
        plt.scatter(warped_cloud0[:,0], warped_cloud0[:,1], c=(0,1,0), edgecolors=(0,1,0), marker=',', s=1)
#         warped_rope_nodes0 = f.transform_points(rope_nodes0)
#         for i, (i_start, i_end) in enumerate(zip(pts_segmentation_inds0[:-1], pts_segmentation_inds0[1:])):
#             color = 'r' if len(pts_segmentation_inds0)<=2 else np.tile(np.array(colorsys.hsv_to_rgb(float(i)/(len(pts_segmentation_inds0)-2),1,1)), (i_end-i_start,1))
#             plt.scatter(warped_rope_nodes0[i_start:i_end,0], warped_rope_nodes0[i_start:i_end,1], c=color, edgecolors=color, marker=',', s=1)
        grid_means = .5 * (rope_nodes0.max(axis=0) + rope_nodes0.min(axis=0))
        grid_mins = grid_means - (rope_nodes0.max(axis=0) - rope_nodes0.min(axis=0))
        grid_maxs = grid_means + (rope_nodes0.max(axis=0) - rope_nodes0.min(axis=0))
        plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=np.median(rope_nodes0, axis=0)[2], xres=.05, yres=.05, draw=False)
    
    plt.draw()

def plot_correspondence(x_nd, y_nd):
    lines = np.array(zip(x_nd, y_nd))
    lc = matplotlib.collections.LineCollection(lines)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()
