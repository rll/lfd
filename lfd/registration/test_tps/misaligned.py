"""
Test tps implementation
"""

import numpy as np
from lfd.registration.tps import ThinPlateSpline, tps_fit3, tps_fit_feedback

def construct_nonworking_point_pairs(d=2):
    if d != 2:
        raise NotImplementedError("todo")
    
    num_pc_points = 8
    half_pc_points = num_pc_points // 2;
    n_pc = 2
    X = np.zeros((num_pc_points * n_pc, d))
    Y = np.zeros((num_pc_points * n_pc, d))
    """"
    x x x x
    x x x x 
        x x x x 
        x x x x
    map to:
    x   x   x   x
    x   x   x   x
          x   x   x   x
          x   x   x   x
    """
    # construct X
    for i in range(half_pc_points):
        X[i,:] = np.array([2*i, 0])
        X[i+half_pc_points,:] = np.array([2*i, 2])
    tmp = np.copy(X[:num_pc_points,:])
    tmp[:, 0] += 4
    X[num_pc_points:,:] = tmp

    # construct Y
    for i in range(half_pc_points):
        Y[i,:] = np.array([4*i, 0])
        Y[i+half_pc_points,:] = np.array([4*i, 2])
    tmp = np.copy(Y[:num_pc_points,:])
    tmp[:, 0] += 6
    Y[num_pc_points:,:] = tmp

    return (X, Y)

def construct_working_point_pairs(d=2):
    if d != 2:
        raise NotImplementedError("todo")
    
    num_pc_points = 8
    half_pc_points = num_pc_points // 2;
    n_pc = 2
    X = np.zeros((num_pc_points * n_pc, d))
    Y = np.zeros((num_pc_points * n_pc, d))

    """"
    x x x x
    x x x x 
        x x x x 
        x x x x
    map to:
    x   x   x   x
    x   x   x   x
            x   x   x   x
            x   x   x   x
    """
    # construct X
    for i in range(half_pc_points):
        X[i,:] = np.array([2*i, 0])
        X[i+half_pc_points,:] = np.array([2*i, 2])
    tmp = np.copy(X[:num_pc_points,:])
    tmp[:, 0] += 4
    X[num_pc_points:,:] = tmp

    # construct Y
    for i in range(half_pc_points):
        Y[i,:] = np.array([4*i, 0])
        Y[i+half_pc_points,:] = np.array([4*i, 2])
    tmp = np.copy(Y[:num_pc_points,:])
    tmp[:, 0] += 8
    Y[num_pc_points:,:] = tmp
    return (X, Y)

def test_old_implementation(X, Y):
    f = ThinPlateSpline(d=2)
    wt_n = None
    bend_coef = 0.0001
    rot_coef = [0.0001, 0.0001]
    theta = tps_fit3(X, Y, bend_coef, rot_coef, wt_n)
    f.update(X, Y, bend_coef, rot_coef, wt_n, theta)
    f_X = f.transform_points(X)
    diff = f_X - Y
    abs_diff = sum(sum(abs(diff)))
    return abs_diff

def test_new_implementation(X, Y):
    f = ThinPlateSpline(d=2)
    wt_n = None
    bend_coef = 0.0001
    rot_coef = [0.0001, 0.0001]
    num_iter = 0
    nu_bd = np.zeros((2, 2))
    tau_bd = np.zeros((2, 2))
    lamb = np.zeros((len(X), 2))
    eta = pow(10, -9) # step size (-9) is not bad
    
    X_combined = np.vstack((X, tau_bd))
    Y_combined = np.vstack((Y, tau_bd))

    f.x_na = X_combined
    f.bend_coef = bend_coef
    f.rot_coef = rot_coef
    while num_iter < 1000: 
        num_iter += 1
        theta = tps_fit_feedback(X, Y, bend_coef, rot_coef, wt_n, lamb, nu_bd, tau_bd)
        f.update_theta(theta) 
        f_X = f.transform_points(X_combined)
        diff = Y_combined - f_X
        diff = diff[:len(X),:]
        step = eta / pow(num_iter, 0.05)
        lamb = lamb - step * diff
    # print("diff {}".format(sum(sum(abs(diff)))))
    abs_diff = sum(sum(abs(diff)))
    return abs_diff

def main():
    """ Test should work pairs first """
    print("==== Old implementation ====")
    X, Y = construct_working_point_pairs(d=2)
    abs_diff = test_old_implementation(X, Y)
    print("diff for working pc: {}".format(abs_diff))
    X, Y = construct_nonworking_point_pairs(d=2)
    abs_diff = test_old_implementation(X, Y)
    print("diff for nonworking pc: {}".format(abs_diff))

    print("==== new implementation ====")
    X, Y = construct_working_point_pairs(d=2)
    abs_diff = test_new_implementation(X, Y)
    print("diff for working pc: {}".format(abs_diff))
    X, Y = construct_nonworking_point_pairs(d=2)
    abs_diff = test_new_implementation(X, Y)
    print("diff for nonworking pc: {}".format(abs_diff))

if __name__ == '__main__':
    main()
    
