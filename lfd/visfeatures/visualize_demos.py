#!/usr/bin/env python

import argparse, h5py, math, numpy as np
from matplotlib import pyplot as plt
import IPython as ipy

WIDTH = 4

def bgr_to_rgb(bgr):
    # Change range to 0-1 by dividing by 255                                    
    # Reverse order from BGR to RGB
    if np.max(bgr) > 1:
        bgr = bgr / 255.0

    rgb = np.zeros((bgr.shape))
    rgb[:,:,0] = bgr[:,:,2]
    rgb[:,:,1] = bgr[:,:,1]
    rgb[:,:,2] = bgr[:,:,0]
    return rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_file', type=str)
    args = parser.parse_args()

    demos = h5py.File(args.demo_file, 'r')

    titles = list(demos.keys())
    titles = titles[64:]
    height = int(math.ceil(len(titles) / float(WIDTH)))

    for i in xrange(len(titles)):
        rgb = bgr_to_rgb(demos[titles[i]]['rgb'][()])
        plt.subplot(height,WIDTH,i+1)
        plt.imshow(rgb)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    demos.close()
    plt.show()


if __name__ == "__main__":
    main()
