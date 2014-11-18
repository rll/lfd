from __future__ import division

from lfd.rapprentice import plotting_openrave

# TODO: rapprentice.plotting_openrave and other openrave plottings should go in this file

def registration_plot_cb(sim, x_nd, y_md, f):
    handles = []
    handles.append(sim.env.plot3(x_nd, 5, (1,0,0)))
    handles.append(sim.env.plot3(y_md, 5, (0,0,1)))
    xwarped_nd = f.transform_points(x_nd)
    handles.append(sim.env.plot3(xwarped_nd, 5, (0,1,0)))
    handles.extend(plotting_openrave.draw_grid(sim.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if sim.viewer:
        sim.viewer.Step()
