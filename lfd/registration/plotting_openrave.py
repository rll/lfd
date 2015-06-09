from __future__ import division

from lfd.rapprentice import plotting_openrave
import numpy as np

# TODO: rapprentice.plotting_openrave and other openrave plottings should go in this file

def registration_plot_cb(sim, x_nd, y_md, f):
    if sim.viewer:
        handles = []
        handles.append(sim.env.plot3(x_nd, 5, (1,0,0)))
        handles.append(sim.env.plot3(y_md, 5, (0,0,1)))
        xwarped_nd = f.transform_points(x_nd)
        handles.append(sim.env.plot3(xwarped_nd, 5, (0,1,0)))
        handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
        sim.viewer.Step()

def registration_plot_cb_2d(sim, x_nd, y_md, f, z):
    if sim.viewer:
        handles = []

        xwarped_nd = f.transform_points(x_nd)
        x_nd = np.hstack((x_nd, z * np.ones((len(x_nd), 1))))
        y_md = np.hstack((y_md, z * np.ones((len(y_md), 1))))
        xwarped_nd = np.hstack((xwarped_nd, z * np.ones((len(xwarped_nd), 1))))
        handles.append(sim.env.plot3(x_nd, 5, (1,0,0)))
        handles.append(sim.env.plot3(y_md, 5, (0,0,1)))
        handles.append(sim.env.plot3(xwarped_nd, 5, (0,1,0)))
        handles.extend(plotting_openrave.draw_grid_2d(sim.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .01, yres = .01, zres = -1.0))
        sim.viewer.Step()
        raw_input("look at plot grid")
