#!/usr/bin/env python

import numpy as np
import openravepy

def plot_clouds(env, pc_seq):
    """
    Plot point cloud sequences
    """
    for pc in pc_seq:
        handles = []
        handles.append(env.sim.env.plot3(points = pc, pointsize=3, colors=[0, 1, 0], drawstyle=1))
        env.sim.viewer.Step()
        raw_input("Look at pc")

