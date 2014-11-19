#!/usr/bin/env python

from __future__ import division

import numpy as np

from lfd.environment.simulation import StaticSimulation
from lfd.environment.simulation_object import BoxSimulationObject
from lfd.registration.registration import TpsRpmRegistrationFactory
from lfd.registration.plotting_openrave import registration_plot_cb
from lfd.demonstration.demonstration import Demonstration, SceneState


np.random.seed(0)

table_height = 0.77
table = BoxSimulationObject("table", [1, 0, table_height-.1], [.85, .85, .1], dynamic=False)

sim = StaticSimulation()
sim.add_objects([table])
sim.create_viewer()

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

plot_cb = lambda i, i_em, x_nd, y_md, xtarg_nd, wt_n, f, corr_nm, rad: registration_plot_cb(sim, x_nd, y_md, f)

reg_factory = TpsRpmRegistrationFactory(demos)
regs = reg_factory.batch_register(test_scene_state, callback=plot_cb)
