"""
Functions for fitting and applying thin plate spline transformations
"""
import numpy as np
import scipy.spatial.distance as ssd
import scipy.optimize as opt
from lfd.util import colorize

VERBOSE = False
ENABLE_SLOW_TESTS = False


def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x


def tps_apply_kernel(distmat, dim):
    """
    if d=2: 
        k(r) = 4 * r^2 log(r)
       d=3:
        k(r) = -r
            
    import numpy as np, scipy.spatial.distance as ssd
    x = np.random.rand(100,2)
    d=ssd.squareform(ssd.pdist(x))
    print np.clip(np.linalg.eigvalsh( 4 * d**2 * log(d+1e-9) ),0,inf).mean()
    print np.clip(np.linalg.eigvalsh(-d),0,inf).mean()
    
    Note the actual coefficients (from http://www.geometrictools.com/Documentation/ThinPlateSplines.pdf)
    d=2: 1/(8*sqrt(pi)) = 0.070523697943469535
    d=3: gamma(-.5)/(16*pi**1.5) = -0.039284682964880184
    """

    if dim==2:       
        return 4 * distmat**2 * np.log(distmat+1e-20)
        
    elif dim ==3:
        return -distmat
    else:
        print 'dim =', dim
        raise NotImplementedError
    
    
def tps_kernel_matrix(x_na):
    dim = x_na.shape[1]
    distmat = ssd.squareform(ssd.pdist(x_na))
    return tps_apply_kernel(distmat,dim)

def tps_kernel_matrix2(x_na, y_ma):
    dim = x_na.shape[1]
    distmat = ssd.cdist(x_na, y_ma)
    return tps_apply_kernel(distmat, dim)

def tps_eval(x_ma, lin_ag, trans_g, w_ng, x_na):
    K_mn = tps_kernel_matrix2(x_ma, x_na).astype(np.float32)
    return np.dot(K_mn, w_ng) + np.dot(x_ma, lin_ag) + trans_g[None,:]

def tps_grad(x_ma, lin_ag, _trans_g, w_ng, x_na):
    _N, D = x_na.shape
    M = x_ma.shape[0]

    assert x_ma.shape[1] == 3
    dist_mn = ssd.cdist(x_ma, x_na,'euclidean')

    grad_mga = np.empty((M,D,D))

    lin_ga = lin_ag.T
    for a in xrange(D):
        diffa_mn = x_ma[:,a][:,None] - x_na[:,a][None,:]
        grad_mga[:,:,a] = lin_ga[None,:,a] - np.dot(nan2zero(diffa_mn/dist_mn),w_ng)
    return grad_mga
    
def tps_nr_grad(x_ma, lin_ag, _trans_g, w_ng, x_na, return_tuple = False):
    """
    gradient of green's strain
    """
    N, D = x_na.shape
    M = x_ma.shape[0]

    assert x_ma.shape[1] == 3
    dists_mn = ssd.cdist(x_ma, x_na,'euclidean')
    diffs_mna = x_ma[:,None,:] - x_na[None,:,:]

    grad_mga = np.empty((M,D,D))
    lin_ga = lin_ag.T
    for a in xrange(D):
        grad_mga[:,:,a] = lin_ga[None,:,a] - np.dot(nan2zero(diffs_mna[:,:,a]/dists_mn),w_ng)

    # m n g a b
    Jw_mngab = - (nan2zero(diffs_mna[:,:,None,:,None]/dists_mn[:,:,None,None,None])) * grad_mga[:,None,:,None,:]
    Jw_mngab = Jw_mngab + Jw_mngab.transpose(0,1,2,4,3)        
    Jw_mabng = Jw_mngab.transpose(0,3,4,1,2)
    Jw = Jw_mabng.reshape(M*D**2,N*D)
    
    Jl_mcgab = np.eye(D)[None,:,None,:,None]*grad_mga[:,None,:,None,:]
    Jl_mcgab = Jl_mcgab + Jl_mcgab.transpose(0,1,2,4,3)
    Jl_mabcg = Jl_mcgab.transpose(0,3,4,1,2)
    Jl = Jl_mabcg.reshape(M*D**2,D*D)
    
    Jt = np.zeros((M*D**2,D))
    
    if return_tuple:
        return Jl, Jt, Jw
    else:
        J = np.c_[Jl, Jt, Jw]
        return J
    
def tps_nr_err(x_ma, lin_ag, trans_g, w_ng, x_na):
    """
    green's strain    
    """
    M,D = x_ma.shape

    grad_mga = tps_grad(x_ma, lin_ag, trans_g, w_ng, x_na)
    err_mab = np.empty((M,D,D))
    for m in xrange(M):
        err_mab[m] = np.dot(grad_mga[m].T, grad_mga[m]) - np.eye(D)
    return err_mab.flatten()

def tps_cost(lin_ag, trans_g, w_ng, x_na, y_ng, bend_coef, K_nn=None, return_tuple=False, wt_n = None):
    """
    XXX doesn't include rotation cost
    """
    D = lin_ag.shape[0]
    if K_nn is None: K_nn = tps_kernel_matrix(x_na)
    if wt_n is None: wt_n = np.ones(len(x_na))
    ypred_ng = np.dot(K_nn, w_ng) + np.dot(x_na, lin_ag) + trans_g[None,:]
    res_cost = (wt_n[:,None] * (ypred_ng - y_ng)**2).sum()
    bend_cost = bend_coef * sum(np.dot(w_ng[:,g], np.dot(K_nn, w_ng[:,g])) for g in xrange(D))
    if return_tuple:
        return res_cost, bend_cost, res_cost + bend_cost
    else:
        return res_cost + bend_cost

def tps_nr_cost_eval(lin_ag, trans_g, w_ng, x_na, y_ng, xnr_ma, bend_coef, nr_coef, K_nn = None, return_tuple=False):
    D = lin_ag.shape[0]
    if K_nn is None: K_nn = tps_kernel_matrix(x_na)
    ypred_ng = np.dot(K_nn, w_ng) + np.dot(x_na, lin_ag) + trans_g[None,:]
    res_cost = ((ypred_ng - y_ng)**2).sum()
    bend_cost = bend_coef * sum(np.dot(w_ng[:,g], np.dot(K_nn, w_ng[:,g])) for g in xrange(D))
    nr_cost = nr_coef * (tps_nr_err(xnr_ma, lin_ag, trans_g, w_ng, x_na)**2).sum()
    if return_tuple:
        return res_cost, bend_cost, nr_cost, res_cost + bend_cost + nr_cost
    else:
        return res_cost + bend_cost + nr_cost

def tps_nr_cost_eval_general(lin_ag, trans_g, w_eg, x_ea, y_ng, nr_ma, bend_coef, nr_coef, K_ee = None, return_tuple=True):
    E,D = x_ea.shape
    N = y_ng.shape[0]
    M = nr_ma.shape[0]
    assert E == N+4*M
    
    K_ee = K_ee or tps_kernel_matrix(x_ea)
    K_ne = K_ee[:N]
    x_na = x_ea[:N]
    
    ypred_ng = np.dot(K_ne, w_eg) + np.dot(x_na, lin_ag) + trans_g[None,:]
    res_cost = ((ypred_ng - y_ng)**2).sum()
    bend_cost = bend_coef * sum(np.dot(w_eg[:,g], np.dot(-K_ee, w_eg[:,g])) for g in xrange(D))
    nr_cost = nr_coef * (tps_nr_err(nr_ma, lin_ag, trans_g, w_eg, x_ea)**2).sum()
    if return_tuple:
        return res_cost, bend_cost, nr_cost, res_cost + bend_cost + nr_cost
    else:
        return res_cost + bend_cost + nr_cost    

def tps_fit(x_na, y_ng, bend_coef, rot_coef, wt_n=None, K_nn = None):
    N,D = x_na.shape
        
    K_nn = tps_kernel_matrix(x_na) if K_nn is None else K_nn
    coef_ratio = bend_coef / rot_coef if rot_coef > 0 else 0
    #if wt_n is None: reg_nn = bend_coef * np.eye(N)    
    #else: reg_nn = np.diag(bend_coef/(wt_n + 1e-6))
    #print wt_n
    
    A = np.zeros((N+D+1, N+D+1))
    
    A[:N, :N] = K_nn

    A.flat[np.arange(0,N)*(N+D+2)] += bend_coef/(wt_n if wt_n is not None else 1)

    A[:N, N:N+D] = x_na
    A[:N, N+D] = 1

    A[N:N+D,:N] = x_na.T
    A[N+D,:N] = 1
    
    A[N:N+D, N:N+D] = coef_ratio*np.eye(D)
    
    B = np.empty((N+D+1, D))
    B[:N] = y_ng
    B[N:N+D] = coef_ratio*np.eye(D)
    B[N+D] = 0
    
    X = np.linalg.solve(A, B)
    w_ng = X[:N,:]
    lin_ag = X[N:N+D,:]
    trans_g = X[N+D,:]
    return lin_ag, trans_g, w_ng
    


    
def solve_eqp1(H, f, A):
    """solve equality-constrained qp
    min tr(x'Hx) + sum(f'x)
    s.t. Ax = 0
    """    
    # print f.shape
    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars
    assert A.shape[1] == n_vars
    n_cnts = A.shape[0]
    
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    # columns of N span the null space
    
    # x = Nz
    # then problem becomes unconstrained minimization .5*z'NHNz + z'Nf
    # NHNz + Nf = 0
    z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    x = N.dot(z)
    
    return x
    
def tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape

    K_nn = tps_kernel_matrix(x_na)
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    WQ = wt_n[:,None] * Q
    QWQ = Q.T.dot(WQ)
    H = QWQ
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)
    
    f = -WQ.T.dot(y_ng)
    f[1:d+1,0:d] -= np.diag(rot_coefs)
    
    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
    
    Theta = solve_eqp1(H,f,A)
    
    return Theta[1:d+1], Theta[0], Theta[d+1:]
    
    
def tps_fit2(x_na, y_ng, bend_coef, rot_coef, wt_n=None):
    if wt_n is not None: raise NotImplementedError

    N,D = x_na.shape 
    _u,_s,_vh = np.linalg.svd(np.c_[x_na, np.ones((N,1))], full_matrices=True)
    N_nq = _u[:,4:] # null of data
    K_nn = tps_kernel_matrix(x_na)
    Q_nn = np.c_[x_na, np.ones((N,1)),K_nn.dot(N_nq)]
    QQ_nn = np.dot(Q_nn.T, Q_nn)
    
    A = QQ_nn    
    A[4:, 4:] += bend_coef * N_nq.T.dot(K_nn).dot(N_nq)
    B = Q_nn.T.dot(y_ng)

    A[:3, :3] += rot_coef * np.eye(3)
    B[:3, :3] += rot_coef * np.eye(3)
    
    X = np.linalg.solve(A,B)

    lin_ag = X[:D,:]
    trans_g = X[D,:]    
    w_ng = N_nq.dot(X[D+1:,:])
    return lin_ag, trans_g, w_ng

def tps_nr_fit(x_na, y_ng, bend_coef, nr_ma, nr_coef, method="newton"):
    N,D = x_na.shape
    lin_ag, trans_g, w_ng = tps_fit2(x_na, y_ng, bend_coef, 1e-3)
    #return lin_ag, trans_g, w_ng

    ##for testing that it takes one step when nonrigidity cost is zero:
    #lin_ag, trans_g, w_ng = tps_fit(x_na, y_ng, bend_coef, 0)
    #res_cost, bend_cost, nr_cost, fval = tps_nr_cost_eval(lin_ag, trans_g, w_ng, x_na, nr_ma, bend_coef, nr_coef, return_tuple=True)
    #print "CORRECT COST, res,bend,nr,total = %.3e, %.3e, %.3e, %.3e"%(res_cost, bend_cost, nr_cost, fval)
    #lin_ag += np.random.randn(*lin_ag.shape)*5
    #res_cost, bend_cost, nr_cost, fval = tps_nr_cost_eval(lin_ag, trans_g, w_ng, x_na, nr_ma, bend_coef, nr_coef, return_tuple=True)
    #print "NOISE ADDED COST, res,bend,nr,total = %.ef, %.3e, %.3e, %.3e"%(res_cost, bend_cost, nr_cost, fval)
    
    _u,_s,_vh = np.linalg.svd(np.c_[x_na, np.ones((N,1))], full_matrices=True)
    N_nq = _u[:,4:] # null of data
    #w_ng = N_nq.dot(N_nq.T.dot(w_ng))
        
    K_nn = tps_kernel_matrix(x_na)
    Q_nn = np.c_[x_na, np.ones((N,1)),K_nn.dot(N_nq)]
    QQ_nn = np.dot(Q_nn.T, Q_nn)
    Bend_nn = np.zeros((N,N))
    Bend_nn[4:, 4:] = - N_nq.T.dot(K_nn.dot(N_nq))
    
    n_iter=60
    for i in xrange(n_iter):
        X_ng = np.r_[lin_ag, trans_g[None,:], N_nq.T.dot(w_ng)]

        res_cost, bend_cost, nr_cost, fval = tps_nr_cost_eval(lin_ag, trans_g, w_ng, x_na, y_ng, nr_ma, bend_coef, nr_coef, return_tuple=True)
        if VERBOSE: print colorize("iteration %i, cost %.3e"%(i, fval), 'red'),
        if VERBOSE: print "= %.3e (res) + %.3e (bend) + %.3e (nr)"%(res_cost, bend_cost, nr_cost)
                
        
        Jl_zcg, Jt_zg, Jw_zng = tps_nr_grad(nr_ma, lin_ag, trans_g, w_ng, x_na, return_tuple=True)
        nrerr_z = tps_nr_err(nr_ma, lin_ag, trans_g, w_ng, x_na)
        
        
        if method == "newton":
            fullstep_ng = np.empty((N,D))
            for g in xrange(D):
                J_zn = np.c_[Jl_zcg[:,g::D], Jt_zg[:,g::D], Jw_zng[:,g::D].dot(N_nq)]
                JJ_nn = np.dot(J_zn.T, J_zn)
                A = nr_coef*JJ_nn + QQ_nn + bend_coef*Bend_nn
                X0 = X_ng[:,g]
                B = nr_coef*np.dot(J_zn.T, np.dot(J_zn, X0) - nrerr_z) + Q_nn.T.dot(y_ng[:,g])
                fullstep_ng[:,g] = np.linalg.solve(A,B) - X_ng[:,g]

        elif method == "gradient":
            # def eval_partial(cand_X_ng):
            #     cand_X_ng = cand_X_ng.reshape(-1,3)
            #     cand_lin_ag, cand_trans_g, cand_w_ng = cand_X_ng[:D], cand_X_ng[D], N_nq.dot(cand_X_ng[D+1:])
            #     fval_cand = tps_nr_cost_eval(cand_lin_ag, cand_trans_g, cand_w_ng, x_na, y_ng, nr_ma, bend_coef, nr_coef)
            #     return fval_cand
            # def eval_partial2(cand_X_ng):
            #     return ((Q_nn.dot(X_ng) - y_ng)**2).sum()
            # def eval_partial3(cand_X_ng):
            #     cand_X_ng = cand_X_ng.reshape(-1,3)
            #     cand_lin_ag, cand_trans_g, cand_w_ng = cand_X_ng[:D], cand_X_ng[D], N_nq.dot(cand_X_ng[D+1:])
            #     return ((y_ng - tps_eval(x_na, cand_lin_ag, cand_trans_g, cand_w_ng, x_na))**2).sum()
            
            
            grad_ng = np.empty((N,D))
            for g in xrange(D-1,-1,-1):
                Jnr_zn = np.c_[Jl_zcg[:,g::D], Jt_zg[:,g::D], Jw_zng[:,g::D].dot(N_nq)]
                grad_ng[:,g] = 2 * nr_coef * nrerr_z.dot(Jnr_zn) \
                    + 2 * Q_nn.T.dot(Q_nn.dot(X_ng[:,g]) - y_ng[:,g]) \
                    + 2 * bend_coef * Bend_nn.dot(X_ng[:,g])

            #assert np.allclose(eval_partial2(X_ng), eval_partial3(X_ng))
            #assert np.allclose(eval_partial(X_ng), eval_partial2(X_ng))
            #grad0_ng = ndt.Gradient(eval_partial)(X_ng.flatten()).reshape(-1,3)
            fullstep_ng = -grad_ng
            #assert np.allclose(grad0_ng, grad_ng)
            
            
            

        cost_improved = False
        for stepsize in 3.**np.arange(0,-10,-1):
            cand_X_ng = X_ng + fullstep_ng*stepsize
            cand_lin_ag, cand_trans_g, cand_w_ng = cand_X_ng[:D], cand_X_ng[D], N_nq.dot(cand_X_ng[D+1:])
            fval_cand = tps_nr_cost_eval(cand_lin_ag, cand_trans_g, cand_w_ng, x_na, y_ng, nr_ma, bend_coef, nr_coef)
            if VERBOSE: print "stepsize: %.1g, fval: %.3e"%(stepsize, fval_cand)
            if fval_cand < fval:
                cost_improved = True
                break
        if not cost_improved:
            if VERBOSE: print "couldn't improve objective"
            break

            
        lin_ag = cand_lin_ag
        trans_g = cand_trans_g
        w_ng = cand_w_ng
    return lin_ag, trans_g, w_ng



def tps_nr_fit_enhanced(x_na, y_ng, bend_coef, nr_ma, nr_coef):
    
    N,D = x_na.shape
    M = nr_ma.shape[0]
    E = N + 4*M
    F = E - M
    Q = N + 3*M - 4
    
    s = .1 # tetrahedron sidelength (meters)
    u = 1/(2*np.sqrt(2))
    
    tetra_pts = []
    for pt in nr_ma:
        tetra_pts.append(s*np.r_[-.5, 0, -u]+pt)
        tetra_pts.append(s*np.r_[+.5, 0, -u]+pt)
        tetra_pts.append(s*np.r_[0, -.5, +u]+pt)
        tetra_pts.append(s*np.r_[0, +.5, +u]+pt)
    
    x_ea = np.r_[x_na, tetra_pts]

    badsub_ex = np.c_[x_ea, np.ones((E,1)), np.r_[np.zeros((N,M)), np.repeat(np.eye(M), 4, axis=0)]]    
    lin_ag, trans_g, w_ng = tps_fit2(x_na, y_ng,  bend_coef, 1e-3)
    w_eg = np.r_[w_ng, np.zeros((4*M, D))]

    assert badsub_ex.shape[0] >= badsub_ex.shape[1]
    _u,_s,_vh = np.linalg.svd(badsub_ex, full_matrices=True)
    assert badsub_ex.shape[1] == _s.size
    N_eq = _u[:,badsub_ex.shape[1]:] # null of data
        
    assert N_eq.shape == (E,Q)

    assert E == N + 4*M
    assert F == Q + 4
    # e is number of kernels
    # q is number of nonrigid dofs
    # f is total number of dofs
    K_ee = tps_kernel_matrix(x_ea)
    K_ne = K_ee[:N, :]
    Q_nf = np.c_[x_na, np.ones((N,1)),K_ne.dot(N_eq)]
    QQ_ff = np.dot(Q_nf.T, Q_nf)
    Bend_ff = np.zeros((F,F))
    Bend_ff[4:, 4:] = - N_eq.T.dot(K_ee.dot(N_eq)) # K_qq
    
    assert Q_nf.shape == (N, F)
    assert w_eg.shape == (E, D)
    
    n_iter=40
    for i in xrange(n_iter):
        
        
        # if plotting and i%plotting==0:
            # import lfd.registration as lr
            # lr.Globals.setup()
            # def eval_partial(x_ma):
            #     return tps_eval(x_ma, lin_ag, trans_g, w_eg, x_ea) 
            # lr.plot_orig_and_warped_clouds(eval_partial, x_na, y_ng, res=.008)            
        
        X_fg = np.r_[lin_ag, 
                    trans_g[None,:], 
                    N_eq.T.dot(w_eg)]

        res_cost, bend_cost, nr_cost, fval = tps_nr_cost_eval_general(lin_ag, trans_g, w_eg, x_ea, y_ng, nr_ma, bend_coef, nr_coef, return_tuple=True)
        if VERBOSE: print colorize("iteration %i, cost %.3e"%(i, fval), 'red'),
        if VERBOSE: print "= %.3e (res) + %.3e (bend) + %.3e (nr)"%(res_cost, bend_cost, nr_cost)
                
        
        Jl_zcg, Jt_zg, Jw_zeg = tps_nr_grad(nr_ma, lin_ag, trans_g, w_eg, x_ea, return_tuple=True)
        nrerr_z = tps_nr_err(nr_ma, lin_ag, trans_g, w_eg, x_ea)        
        
        fullstep_fg = np.empty((F,D))
        for g in xrange(D):
            J_zf = np.c_[Jl_zcg[:,g::D], Jt_zg[:,g::D], Jw_zeg[:,g::D].dot(N_eq)]
            JJ_ff = np.dot(J_zf.T, J_zf)
            A_ff = nr_coef*JJ_ff + QQ_ff + bend_coef*Bend_ff
            X0 = X_fg[:,g]
            B_f = nr_coef*np.dot(J_zf.T, np.dot(J_zf, X0) - nrerr_z) + Q_nf.T.dot(y_ng[:,g])
            fullstep_fg[:,g] = np.linalg.solve(A_ff,B_f) - X_fg[:,g]

        cost_improved = False
        for stepsize in 3.**np.arange(0,-10,-1):
            cand_X_fg = X_fg + fullstep_fg*stepsize
            cand_lin_ag, cand_trans_g, cand_w_eg = cand_X_fg[:D], cand_X_fg[D], N_eq.dot(cand_X_fg[D+1:])
            fval_cand = tps_nr_cost_eval_general(cand_lin_ag, cand_trans_g, cand_w_eg, x_ea, y_ng, nr_ma, bend_coef, nr_coef, return_tuple=False)
            if VERBOSE: print "stepsize: %.1g, fval: %.3e"%(stepsize, fval_cand)
            if fval_cand < fval:
                cost_improved = True
                break
        if not cost_improved:
            if VERBOSE: print "couldn't improve objective"
            break

            
        lin_ag = cand_lin_ag
        trans_g = cand_trans_g
        w_eg = cand_w_eg
    return lin_ag, trans_g, w_eg, x_ea


def tps_fit_fixedrot(x_na, y_ng, bend_coef, lin_ag, K_nn = None, wt_n=None):
    """
    minimize (Y-KA-XB-1C)'W(Y-KA-XB-1C) + tr(A'KA) + r(B)
    """
    if wt_n is not None: raise NotImplementedError
    
    N,_D = x_na.shape
    _u,_s,_vh = np.linalg.svd(np.c_[x_na, np.ones((N,1))], full_matrices=True)
    N_nq = _u[:,4:] # null of data
    K_nn = tps_kernel_matrix(x_na)

    Q_nn = np.c_[np.ones((N,1)),K_nn.dot(N_nq)]
    QQ_nn = np.dot(Q_nn.T, Q_nn)
    
    A = QQ_nn
    A[1:, 1:] += bend_coef * N_nq.T.dot(K_nn).dot(N_nq)
    B = Q_nn.T.dot(y_ng-x_na.dot(lin_ag))

    X = np.linalg.solve(A,B)

    trans_g = X[0,:]    
    w_ng = N_nq.dot(X[1:,:])
    return trans_g, w_ng
    

def tps_fit_regrot(x_na, y_ng, bend_coef, rfunc, wt_n=None, max_iter = 1, inner_max_iter=100, rgrad=None, l_init=None):
    """
    minimize (Y-KA-XB-1C)' W (Y-KA-XB-1C) + tr(A'KA) + r(B)
    subject to A'(X 1) = 0
    """
    if rgrad is not None: raise NotImplementedError
    if wt_n is not None: raise NotImplementedError


    N,_D = x_na.shape
    K_nn = tps_kernel_matrix(x_na)
    # initialize with tps_fit and small rotation regularization
    if l_init is None: 
        lin_ag, trans_g, w_ng = tps_fit3(x_na, y_ng, bend_coef, .01, wt_n)
    else:
        lin_ag = l_init
        if True: print "initializing rotation with\n ",lin_ag
        trans_g, w_ng = tps_fit_fixedrot(x_na, y_ng, bend_coef, lin_ag, K_nn, wt_n)
    #w_ng *= 0
    Q_nn = np.eye(N) - np.outer(np.ones(N), np.ones(N))/N
    for _ in xrange(max_iter):
        e_ng = y_ng - K_nn.dot(w_ng)
        xQe_ag = x_na.T.dot(Q_nn).dot(e_ng)
        xQx_aa = x_na.T.dot(Q_nn).dot(x_na)
        eQe_g = ((e_ng - e_ng.mean(axis=0))**2).sum(axis=0)
        def f(x): 
            b_ag = x.reshape(3,3)
            out = sum([b_ag[:,i].T.dot(xQx_aa).dot(b_ag[:,i]) - 2*b_ag[:,i].T.dot(xQe_ag[:,i]) + eQe_g[i] for i in xrange(3)]) + rfunc(b_ag)
            return out
        x0 = lin_ag.flatten()
        (soln, fopt,_,_,_,_,allsolns) = opt.fmin_powell(f, x0, maxiter=inner_max_iter, disp=bool(VERBOSE), retall=True, full_output=True)
        if not np.isfinite(soln).all(): 
            print "warning, optimization gave infinite result"
            soln = allsolns[-2]
        if not np.isfinite(fopt):
            soln = lin_ag
            print "warning, optimization gave fopt=infinity"
        lin_ag = soln.reshape(3,3)
        assert np.isfinite(lin_ag).all()
        trans_g, w_ng = tps_fit_fixedrot(x_na, y_ng, bend_coef, lin_ag, K_nn, wt_n)
    #trans_g = y_ng.mean(axis=0) - tps_eval(x_na, lin_ag, np.zeros(3), w_ng, x_na).mean(axis=0)
    if VERBOSE: print "rotation result", lin_ag
    return lin_ag, trans_g, w_ng

#def tps_fit_regrot2(x_na, y_ng, bend_coef, rfunc, wt_n=None, max_iter = 20):
    #"""
    #minimize (Y-KA-XB-1C)' W (Y-KA-XB-1C) + tr(A'KA) + r(B)
    #subject to A'(X 1) = 0
    #"""
    #K_nn = ssd.squareform(ssd.pdist(x_na))
    #N,_ = x_na.shape
    #def f(x):
        #x = x.reshape(N+4,3)
        #lin_ag = x[:3]
        #trans_g = x[3]
        #w_ng = x[4:]
        #return tps_cost_regrot(lin_ag, trans_g, w_ng, x_na, y_ng, bend_coef, rfunc, K_nn, wt_n)
    #lin_ag, trans_g, w_ng = tps_fit2(x_na, y_ng, bend_coef, .01, wt_n)
    #w_ng *= 0
    #xopt = opt.fmin_powell(f, np.r_[lin_ag, trans_g[None,:], w_ng].flatten(), maxiter=max_iter)
    #lin_ag = xopt[:3]
    #trans_g = xopt[3]
    #w_ng = xopt[4:]
    #return lin_ag, trans_g, w_ng


def tps_cost_regrot(lin_ag, trans_g, w_ng, x_na, y_ng, bend_coef, rfunc, K_nn = None, wt_n=None):
    """
    (Y-KA-XB-1C)' W (Y-KA-XB-1C) + tr(A'KA) + r(B)
    subject to A'(X 1) = 0
    """
    if wt_n is not None: raise NotImplementedError
    return tps_cost(lin_ag, trans_g, w_ng, x_na, y_ng, bend_coef, K_nn) + rfunc(lin_ag)
