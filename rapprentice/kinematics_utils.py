import numpy as np
import scipy.interpolate as si
from numpy import pi
from rapprentice import math_utils as mu

def smaller_ang(x):
    return (x + pi)%(2*pi) - pi
def closer_ang(x,a,dir=0):
    """                                                                        
    find angle y (==x mod 2*pi) that is close to a                             
    dir == 0: minimize absolute value of difference                            
    dir == 1: y > x                                                            
    dir == 2: y < x                                                            
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*pi)
    elif dir == -1:
        return a + (x-a)%(2*pi) - 2*pi

def closer_joint_angles(pos,seed):
    result = np.array(pos)
    for i in [2,4,6]:
        result[i] = closer_ang(pos[i],seed[i],0)
    return result


def get_velocities(positions, times, tol):
    positions = np.atleast_2d(positions)
    n = len(positions)
    deg = min(3, n - 1)
    
    good_inds = np.r_[True,(abs(times[1:] - times[:-1]) >= 1e-6)]
    good_positions = positions[good_inds]
    good_times = times[good_inds]
    
    if len(good_inds) == 1:
        return np.zeros(positions[0:1].shape)
    
    (tck, _) = si.splprep(good_positions.T,s = tol**2*(n+1), u=good_times, k=deg)
    #smooth_positions = np.r_[si.splev(times,tck,der=0)].T
    velocities = np.r_[si.splev(times,tck,der=1)].T    
    return velocities

def smooth_positions(positions, tol):
    times = np.arange(len(positions))
    positions = np.atleast_2d(positions)
    n = len(positions)
    deg = min(3, n - 1)
    
    good_inds = np.r_[True,(abs(times[1:] - times[:-1]) >= 1e-6)]
    good_positions = positions[good_inds]
    good_times = times[good_inds]
    
    if len(good_inds) == 1:
        return np.zeros(positions[0:1].shape)
    
    (tck, _) = si.splprep(good_positions.T,s = tol**2*(n+1), u=good_times, k=deg)
    smooth_positions = np.r_[si.splev(times,tck,der=0)].T
    return smooth_positions

def unif_resample(x,n,weights,tol=.001,deg=3):    
    x = np.atleast_2d(x)
    weights = np.atleast_2d(weights)
    x = mu.remove_duplicate_rows(x)
    x_scaled = x * weights
    dl = mu.norms(x_scaled[1:] - x_scaled[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    (tck,_) = si.splprep(x_scaled.T,k=deg,s = tol**2*len(x),u=l)
    
    newu = np.linspace(0,l[-1],n)
    out_scaled = np.array(si.splev(newu,tck)).T
    out = out_scaled/weights
    return out