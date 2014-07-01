import numpy as np
import tps, registration
from registration import tps_rpm_bij
from registration import Transformation, ThinPlateSpline # classes need to be imported this way in order to be defined properly in the cluster
import pp
import IPython as ipy

def tps_rpm_bij_grid(x_knd, y_lmd, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
                     plotting = False, plot_cb = None, x_weights = None, y_weights = None, outlierprior = .1, outlierfrac = 2e-1, 
                     parallel = False, ppservers=(), partition_step=None):
    """
    If parallel=True the computation is split among the cores of the local machine and the cores of the cluster
    So, if parallel=True and ppservers=(), the computation is split only among the local cores
    If parallel=True and partition_step is especified, the computation is split in blocks of partition_step x partition_step
    """
    #TODO warn if default parameters are different?
    if len(x_knd.shape) == 2:
        x_knd.resize((1,)+x_knd.shape)
    if len(y_lmd.shape) == 2:
        y_lmd.resize((1,)+y_lmd.shape)
    k = x_knd.shape[0]
    l = y_lmd.shape[0]
    tps_tups = np.empty((k,l), dtype=object)
     
    if not parallel:
        for i in range(k):
            for j in range(l):
                tps_tups[i,j] = tps_rpm_bij(x_knd[i], y_lmd[j], n_iter=n_iter, reg_init=reg_init, reg_final=reg_final, 
                                            rad_init=rad_init, rad_final=rad_final, rot_reg=rot_reg, 
                                            plotting=plotting, plot_cb=plot_cb, x_weights=x_weights, y_weights=y_weights, 
                                            outlierprior=outlierprior, outlierfrac=outlierfrac)
    else:
        job_server = pp.Server(ppservers=ppservers)
        #TODO check if servers on nodes are running
        print "Starting pp with", job_server.get_ncpus(), "local workers"
 
        # make sure to change the order of optional arguments if the function's signature is changed
        # parallel=False
        opt_arg_vals = (n_iter, reg_init, reg_final, rad_init, rad_final, rot_reg, plotting, plot_cb, x_weights, y_weights, outlierprior, outlierfrac, False, ppservers)
        dep_funcs = ()
        modules = ("import numpy as np",
                   "import scipy.spatial.distance as ssd")
        myglobals = {'tps_rpm_bij':registration.tps_rpm_bij, 
                 'loglinspace':registration.loglinspace,
                 'Transformation':Transformation,
                 'ThinPlateSpline':ThinPlateSpline,
                 'tps_eval':tps.tps_eval,
                 'tps_kernel_matrix2':tps.tps_kernel_matrix2,
                 'tps_apply_kernel':tps.tps_apply_kernel,
                 'balance_matrix3':registration.balance_matrix3,
                 'fit_ThinPlateSpline':registration.fit_ThinPlateSpline,
                 'tps_fit3':tps.tps_fit3,
                 'tps_kernel_matrix':tps.tps_kernel_matrix,
                 'solve_eqp1':tps.solve_eqp1,
                 'tps_cost':tps.tps_cost
                 }
        if partition_step:
            partitions = [(i,min(i+partition_step,k),j,min(j+partition_step,l)) for i in range(0,k,partition_step) for j in range(0,l,partition_step)]
        else:
            partitions = [(i,i+1,0,l) for i in range(k)]
        jobs = [job_server.submit(tps_rpm_bij_grid, (x_knd[i_start:i_end], y_lmd[j_start:j_end])+opt_arg_vals, dep_funcs, modules, globals=myglobals) 
                for (i_start,i_end,j_start,j_end) in partitions]
        for ((i_start,i_end,j_start,j_end),job) in zip(partitions, jobs):
            tps_tups[i_start:i_end,j_start:j_end] = job()
        job_server.print_stats()
     
    return tps_tups
