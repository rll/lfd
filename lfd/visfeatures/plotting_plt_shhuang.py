"""
Plotting functions using matplotlib
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

from lfd.rapprentice import plotting_plt

def plot_tps_registration(x_nd, y_md, f, res = (.1, .1, .04), x_color=None, y_color=None, proj_2d=False, z_intercept=0, x_labels=None, y_labels=None, label_colors=None, save_filename=""):
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
            plot_tps_registration_proj_2d(x_nd, y_md, f, res, x_color, y_color, xwarped_color, z_intercept=z_intercept, x_labels=x_labels, y_labels=y_labels, label_colors=label_colors, save_filename=save_filename)
        else:
            plotting_plt.plot_tps_registration_3d(x_nd, y_md, f, res, x_color, y_color, xwarped_color)
    else:
        plotting_plt.plot_tps_registration_2d(x_nd, y_md, f, x_color, y_color, xwarped_color)

def plot_tps_registration_proj_2d(x_nd, y_md, f, res, x_color, y_color, xwarped_color, z_intercept=0, x_labels=None, y_labels=None, label_colors=None, save_filename=""):
    # set interactive
    plt.ion()
    
    fig = plt.figure('2d projection plot')
    fig.clear()

    x_colors = None
    y_colors = None
    if x_labels != None and label_colors != None:
        x_colors = []
        for i in range(len(x_nd)):
            label = x_labels[i]
            x_colors.append(label_colors[label])
    if y_labels != None and label_colors != None:
        y_colors = []
        for i in range(len(y_md)):
            label = y_labels[i]
            y_colors.append(label_colors[label])
    
    plt.subplot(221, aspect='equal')
    if x_colors == None:
        plt.scatter(x_nd[:,0], x_nd[:,1], c=x_color, edgecolors=x_color, marker=',', s=5)
    else:
        plt.scatter(x_nd[:,0], x_nd[:,1], c=x_colors, edgecolors=x_colors, marker=',', s=5)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    x_median = np.median(x_nd, axis=0)
    plotting_plt.plot_warped_grid_proj_2d(lambda xyz: xyz, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    
    plt.subplot(222, aspect='equal')
    plt.scatter(y_md[:,0], y_md[:,1], c=y_color, marker='+', s=50)
    xwarped_nd = f.transform_points(x_nd)
    plt.scatter(xwarped_nd[:,0], xwarped_nd[:,1], edgecolors=xwarped_color, facecolors='none', marker='o', s=50)
    plot2_axis = plt.axis()

    plt.subplot(223, aspect='equal')
    if y_colors == None:
        plt.scatter(y_md[:,0], y_md[:,1], c=y_color, edgecolors=y_color, marker=',', s=5)
    else:
        plt.scatter(y_md[:,0], y_md[:,1], c=y_colors, edgecolors=y_colors, marker=',', s=5)

    plt.subplot(224, aspect='equal')
    plotting_plt.plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0]/4, yres=res[1]/4, draw=False)
    #plotting_plt.plot_warped_grid_3d(f.transform_points, grid_mins, grid_maxs, xres=res[0], yres=res[1], zres=res[2], draw=False)
    plt.axis(plot2_axis)
    
    plt.draw()
    if len(save_filename) > 0:  # save plot to file
        plt.savefig(save_filename + '.pdf')
    else:
        plt.show()
