#!/usr/bin/env python

from __future__ import division

import numpy as np
import trajoptpy
from core.simulation import StaticSimulation
from core.simulation_object import BoxSimulationObject
from registration.registration import TpsRpmRegistrationFactory
from registration.plotting_openrave import registration_plot_cb

from core.demonstration import Demonstration, SceneState
from registration import solver

np.random.seed(0)

table_height = 0.77
table = BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False)

sim = StaticSimulation()
sim.add_objects([table])

viewer = trajoptpy.GetViewer(sim.env)
camera_matrix = np.array([[ 0,    1, 0,   0],
                          [-1,    0, 0.5, 0],
                          [ 0.5,  0, 1,   0],
                          [ 2.25, 0, 4.5, 1]])
viewer.SetWindowProp(2560,0,1500,1500)
viewer.SetCameraManipulatorMatrix(camera_matrix)

def generate_cloud(x_center_pert=0, max_noise=0.02):
    # generates 40 cm by 60 cm cloud with optional pertubation along the x-axis
    grid = np.array(np.meshgrid(np.linspace(-.2,.2,21), np.linspace(-.3,.3,31))).T.reshape((-1,2))
    grid = np.c_[grid, np.zeros(len(grid))] + np.array([.5, 0, table_height+max_noise])
    cloud = grid + x_center_pert * np.c_[(0.3 - np.abs(grid[:,1]-0))/0.3, np.zeros((len(grid),2))] + (np.random.random((len(grid), 3)) - 0.5) * 2 * max_noise
    return cloud

demos = {}
for x_center_pert in np.arange(-0.1, 0.6, 0.1):
    demo_name = "demo_{}".format(x_center_pert)
    demo_cloud = generate_cloud(x_center_pert=x_center_pert)
    demo_scene_state = SceneState(demo_cloud, downsample_size=0.025)
    demo = Demonstration(demo_name, demo_scene_state, None)
    demos[demo_name] = demo

test_cloud = generate_cloud(x_center_pert=0.2)
test_scene_state = SceneState(test_cloud, downsample_size=0.025)

reg_factory = TpsRpmRegistrationFactory({}, f_solver_factory=solver.TpsSolverFactory())
plot_cb = lambda *args: registration_plot_cb(sim, *args)
for demo in demos.values():
    reg = reg_factory.register(demo, test_scene_state, plotting=True, plot_cb=plot_cb)
