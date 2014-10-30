tps-opt
=======

High-Throughput Library for Fitting Thin Plate Splines

Dependencies
============
python2.7, scipy0.14, numpy1.8.1, gfortran, cuda6.0, PyCuda2013.1.1, scikits.cuda0.5.0, cmake, boost-python

Install Instructions
====================
You can install PyCuda, numpy and scipy with pip install to get the latest versions.

Install Cuda6.0
http://www.r-tutor.com/gpu-computing/cuda-installation/cuda6.0-ubuntu
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html

Install latest scikits.cuda from source (version available through pip doesn't have integration for the batched cublas calls yet).
```
$ git clone https://github.com/lebedov/scikits.cuda.git
$ cd scikits.cuda
$ python setup.py install 
```
The last line may need to run as root. CUDA path may need to be set.

Build the additional cuda functionality
```
$ cd lfd/tpsopt
$ cmake .
$ make
```
It has been tested with the RLL overhand-knot tying demonstration dataset. Obtain a copy from https://www.dropbox.com/s/wnt3j42jp5solr8/actions.h5. 

To check the build, cd to tps-opt/tpsopt and run the appropriate version of the following
```
dhm@primus:~$ cd src/tps-opt/tpsopt/
dhm@primus:~/src/tps-opt/tpsopt$ python precompute.py path/to/actions.h5 --replace --verbose --fill_traj
precomputed tps solver for segment failuretwo_5-seg02
dhm@primus:~/src/tps-opt/tpsopt$ python batchtps.py --input_file path/to/actions.h5 --test_full
running basic unit tests
UNIT TESTS PASSED
unit tests passed, doing full check on batch tps rpm
testing source cloud 147
tests succeeded!
dhm@primus:~/src/tps-opt/tpsopt$ python batchtps.py --input_file path/to/actions.h5
batchtps initialized
Running Timing test 99/100
Timing Tests Complete
Batch Size:                     148
Mean Compute Time per Batch:    0.0724914503098
BiDirectional TPS fits/second:  2041.62007199
```
You should see results analogous to those above. That example was run with an NVIDIA GTX770.
You can set the default behavior for the batchtps main in batchtps.parse_arguments.
The default parameters for the TPS fits is found in defaults.py. 

Notes
=====
In the scripts directory is Robert Kern's kernprof.py. It is a line-by-line profilier for python. The required packages can be installed from https://pythonhosted.org/line_profiler/. Running batchtps through that with --sync will let you see a (slower due to profiler overhead, and synchronous execution) breakdown of the timings for the gpu kernel calls. To run this, you will need to comment in the @profile decorators on the functions you wish to time. Example output for timing batch_tps_rpm and the various functions calls is included below. If the library is running slowly, check your results against that as a first step to gauge where the problem is.

Example Line-By-Line Timings
============================
Example output of kernprof.py with profiling enabled for core functionality.
```
dhm@primus:~/src/tps-opt/tpsopt$ python ../scripts/kernprof.py -lv batchtps.py --sync
batchtps initialized
Running Timing test 99/100
Timing Tests Complete
Batch Size:			148
Mean Compute Time per Batch:	0.169496803284
BiDirectional TPS fits/second:	873.172809945
Wrote profile results to batchtps.py.lprof
Timer unit: 1e-06 s

File: batchtps.py
Function: transform_points at line 260
Total time: 5.82494 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   260                                               @profile
   261                                               def transform_points(self):
   262                                                   """
   263                                                   computes the warp of self.pts under the current tps params
   264                                                   """
   265      2200        23531     10.7      0.4          fill_mat(self.pt_w_ptrs, self.trans_d_ptrs, self.dims_gpu, self.N)
   266      2200         1984      0.9      0.0          dot_batch_nocheck(self.pts,         self.lin_dd,      self.pts_w,
   267      2200      2590108   1177.3     44.5                            self.pt_ptrs,     self.lin_dd_ptrs, self.pt_w_ptrs) 
   268      2200         1994      0.9      0.0          dot_batch_nocheck(self.kernels,     self.w_nd,        self.pts_w,
   269      2200      2490537   1132.1     42.8                            self.kernel_ptrs, self.w_nd_ptrs,   self.pt_w_ptrs) 
   270      2200       716784    325.8     12.3          sync()

File: batchtps.py
Function: get_target_points at line 272
Total time: 3.30969 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   272                                               @profile
   273                                               def get_target_points(self, other, outlierprior=1e-1, outlierfrac=1e-2, outliercutoff=1e-2, 
   274                                                                     T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
   275                                                   """
   276                                                   computes the target points for self and other
   277                                                   using the current warped points for both                
   278                                                   """
   279      1000          628      0.6      0.0          init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
   280      1000          502      0.5      0.0                       self.pt_w_ptrs, other.pt_w_ptrs, 
   281      1000          466      0.5      0.0                       self.dims_gpu, other.dims_gpu,
   282      1000          452      0.5      0.0                       self.N, outlierprior, outlierfrac, T, 
   283      1000        14230     14.2      0.4                       self.corr_cm_ptrs, self.corr_rm_ptrs)
   284      1000      1305172   1305.2     39.4          sync()
   285      1000          681      0.7      0.0          norm_prob_nm(self.corr_cm_ptrs, self.corr_rm_ptrs, 
   286      1000          555      0.6      0.0                       self.dims_gpu, other.dims_gpu, self.N, outlierfrac, norm_iters,
   287      1000        11178     11.2      0.3                       self.r_coeff_ptrs, self.c_coeff_rn_ptrs, self.c_coeff_cn_ptrs)        
   288      1000      1629934   1629.9     49.2          sync()
   289      1000          736      0.7      0.0          get_targ_pts(self.pt_ptrs, other.pt_ptrs,
   290      1000          514      0.5      0.0                       self.pt_w_ptrs, other.pt_w_ptrs,
   291      1000          493      0.5      0.0                       self.corr_cm_ptrs, self.corr_rm_ptrs,
   292      1000          505      0.5      0.0                       self.r_coeff_ptrs, self.c_coeff_rn_ptrs, self.c_coeff_cn_ptrs,
   293      1000          485      0.5      0.0                       self.dims_gpu, other.dims_gpu, 
   294      1000          425      0.4      0.0                       outliercutoff, self.N,
   295      1000        18223     18.2      0.6                       self.pt_t_ptrs, other.pt_t_ptrs)
   296      1000       324514    324.5      9.8          sync()

File: batchtps.py
Function: update_transform at line 298
Total time: 4.44789 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   298                                               @profile
   299                                               def update_transform(self, b):
   300                                                   """
   301                                                   computes the TPS associated with the current target pts
   302                                                   """
   303      2000      1395651    697.8     31.4          self.set_tps_params(self.offset_mats[b])
   304      2000         3255      1.6      0.1          dot_batch_nocheck(self.proj_mats[b],     self.pts_t,     self.tps_params,
   305      2000      2394785   1197.4     53.8                            self.proj_mat_ptrs[b], self.pt_t_ptrs, self.tps_param_ptrs)
   306      2000       654196    327.1     14.7          sync()

File: batchtps.py
Function: mapping_cost at line 307
Total time: 0.532049 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   307                                               @profile
   308                                               def mapping_cost(self, other, bend_coeff=DEFAULT_LAMBDA[1], outlierprior=1e-1, outlierfrac=1e-2, 
   309                                                                  outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
   310                                                   """
   311                                                   computes the error in the current mapping
   312                                                   assumes that the target points have already been filled
   313                                                   """
   314       100       263300   2633.0     49.5          self.transform_points()
   315       100       254329   2543.3     47.8          other.transform_points()
   316       100           69      0.7      0.0          sums = []
   317       100         1119     11.2      0.2          sq_diffs(self.pt_w_ptrs, self.pt_t_ptrs, self.warp_err, self.N, True)
   318       100          616      6.2      0.1          sq_diffs(other.pt_w_ptrs, other.pt_t_ptrs, self.warp_err, self.N, False)
   319       100         7950     79.5      1.5          warp_err = self.warp_err.get()
   320       100         4666     46.7      0.9          return np.sum(warp_err, axis=1)

File: batchtps.py
Function: bending_cost at line 321
Total time: 0.59744 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   321                                               @profile
   322                                               def bending_cost(self, b=DEFAULT_LAMBDA[1]):
   323                                                   ## b * w_nd' * K * w_nd
   324                                                   ## use pts_w as temporary storage
   325       200          209      1.0      0.0          dot_batch_nocheck(self.kernels,     self.w_nd,      self.pts_w,
   326       200          122      0.6      0.0                            self.kernel_ptrs, self.w_nd_ptrs, self.pt_w_ptrs,
   327       200       222393   1112.0     37.2                            b = 0)
   328                                           
   329       200          204      1.0      0.0          dot_batch_nocheck(self.pts_w,     self.w_nd,      self.bend_res,
   330       200          139      0.7      0.0                            self.pt_w_ptrs, self.w_nd_ptrs, self.bend_res_ptrs,
   331       200       236069   1180.3     39.5                            transa='T', b = 0)
   332       200         9957     49.8      1.7          bend_res = self.bend_res_mat.get()        
   333     29800       128347      4.3     21.5          return b * np.array([np.trace(bend_res[i*DATA_DIM:(i+1)*DATA_DIM]) for i in range(self.N)])

File: batchtps.py
Function: bidir_tps_cost at line 334
Total time: 1.15506 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   334                                               @profile
   335                                               def bidir_tps_cost(self, other, bend_coeff=1, outlierprior=1e-1, outlierfrac=1e-2, 
   336                                                                  outliercutoff=1e-2,  T = 5e-3, norm_iters = DEFAULT_NORM_ITERS):
   337       100         4773     47.7      0.4          self.reset_warp_err()
   338       100       533159   5331.6     46.2          mapping_err  = self.mapping_cost(other, outlierprior, outlierfrac, outliercutoff, T, norm_iters)
   339       100       311243   3112.4     26.9          bending_cost = self.bending_cost(bend_coeff)
   340       100       305676   3056.8     26.5          bending_cost += other.bending_cost(bend_coeff)
   341       100          208      2.1      0.0          return mapping_err + bending_cost

File: batchtps.py
Function: set_cld at line 559
Total time: 2.43952 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   559                                               @profile
   560                                               def set_cld(self, cld):
   561                                                   """
   562                                                   sets the cloud for this appropriately
   563                                                   won't allocate any new memory
   564                                                   """                          
   565       101      1862412  18439.7     76.3          proj_mats, offset_mats, K = self.get_sol_params(cld)
   566       101        10860    107.5      0.4          K_gpu = gpu_pad(K, (MAX_CLD_SIZE, MAX_CLD_SIZE))
   567       101         6134     60.7      0.3          cld_gpu = gpu_pad(cld, (MAX_CLD_SIZE, DATA_DIM))
   568     15049        10474      0.7      0.4          self.pts         = [cld_gpu for _ in range(self.N)]
   569     15049         9894      0.7      0.4          self.kernels     = [K_gpu for _ in range(self.N)]
   570       101           75      0.7      0.0          proj_mats_gpu    = dict([(b, gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE)))
   571      1111       114060    102.7      4.7                                   for b, p in proj_mats.iteritems()])
   572       101           73      0.7      0.0          self.proj_mats   = dict([(b, [p for _ in range(self.N)])
   573    150591        94907      0.6      3.9                                   for b, p in proj_mats_gpu.iteritems()])
   574       101          103      1.0      0.0          offset_mats_gpu  = dict([(b, gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM))) 
   575      1111        86652     78.0      3.6                                   for b, p in offset_mats.iteritems()])
   576       101           67      0.7      0.0          self.offset_mats = dict([(b, [p for _ in range(self.N)])
   577    150591        94903      0.6      3.9                                   for b, p in offset_mats_gpu.iteritems()])
   578       101          158      1.6      0.0          self.dims        = [cld.shape[0]]
   579                                           
   580       101        60076    594.8      2.5          self.pt_ptrs.fill(int(self.pts[0].gpudata))
   581       101         1971     19.5      0.1          self.kernel_ptrs.fill(int(self.kernels[0].gpudata))
   582       101        55545    550.0      2.3          self.dims_gpu.fill(self.dims[0])
   583      1111         1122      1.0      0.0          for b in self.bend_coeffs:
   584      1010        15086     14.9      0.6              self.proj_mat_ptrs[b].fill(int(self.proj_mats[b][0].gpudata))
   585      1010        14945     14.8      0.6              self.offset_mat_ptrs[b].fill(int(self.offset_mats[b][0].gpudata))

File: batchtps.py
Function: batch_tps_rpm_bij at line 668
Total time: 14.4213 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   668                                           @profile
   669                                           def batch_tps_rpm_bij(src_ctx, tgt_ctx, T_init = 1e-1, T_final = 5e-3, 
   670                                                                 outlierfrac = 1e-2, outlierprior = 1e-1, outliercutoff = 1e-2, em_iter = EM_ITER_CHEAP):
   671                                               """
   672                                               computes tps rpm for the clouds in src and tgt in batch
   673                                               TODO: Fill out comment cleanly
   674                                               """
   675                                               ##TODO: add check to ensure that src_ctx and tgt_ctx are formatted properly
   676       100          143      1.4      0.0      n_iter = len(src_ctx.bend_coeffs)
   677       100         3633     36.3      0.0      T_vals = loglinspace(T_init, T_final, n_iter)
   678                                           
   679       100        78329    783.3      0.5      src_ctx.reset_tps_params()
   680       100        71218    712.2      0.5      tgt_ctx.reset_tps_params()
   681      1100         1550      1.4      0.0      for i, b in enumerate(src_ctx.bend_coeffs):
   682      1000          996      1.0      0.0          T = T_vals[i]
   683      2000         1855      0.9      0.0          for _ in range(em_iter):
   684      1000      2732661   2732.7     18.9              src_ctx.transform_points()
   685      1000      2591062   2591.1     18.0              tgt_ctx.transform_points()
   686      1000      3324074   3324.1     23.0              src_ctx.get_target_points(tgt_ctx, outlierprior, outlierfrac, outliercutoff, T)
   687      1000      2346748   2346.7     16.3              src_ctx.update_transform(b)
   688                                                       # check_update(src_ctx, b)
   689      1000      2113210   2113.2     14.7              tgt_ctx.update_transform(b)
   690       100      1155837  11558.4      8.0      return src_ctx.bidir_tps_cost(tgt_ctx)
```
