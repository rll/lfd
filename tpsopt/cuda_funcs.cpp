#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include <iostream>
#include "tps.cuh"

namespace py = boost::python;

void pyFloatPrintArr(py::object x, int N){
  float* p = getGPUPointer<float>(x);
  printf("x ptr is %li\n", (long int) p);
  gpuPrintArr(p, N);
}

void pyIntPrintArr(py::object x, int N){
  int* p = getGPUPointer<int>(x);
  printf("x ptr is %li\n", (long int) p);
  gpuPrintArr(p, N);
}

void pyFillMat(py::object dest, py::object val, py::object dims, int N){
  float** dest_ptr = getGPUPointer<float*>(dest);
  float** val_ptr  = getGPUPointer<float*>(val);
  int* dims_ptr    = getGPUPointer<int>(dims);

  fillMat(dest_ptr, val_ptr, dims_ptr, N);
}

void pySqDiffMat(py::object x, py::object y, py::object z, int N, bool overwrite){
  float** x_ptr   = getGPUPointer<float*>(x);
  float** y_ptr   = getGPUPointer<float*>(y);
  float* z_ptr    = getGPUPointer<float>(z);
  
  sqDiffMat(x_ptr, y_ptr, z_ptr, N, overwrite);
}

void pyClosestPointCost(py::object x, py::object y, py::object xdims, py::object ydims, py::object res, int N){
  float** x_ptr = getGPUPointer<float*>(x);
  float** y_ptr = getGPUPointer<float*>(y);

  int* xdims_ptr = getGPUPointer<int>(xdims);
  int* ydims_ptr = getGPUPointer<int>(ydims);
  
  float* res_ptr = getGPUPointer<float>(res);

  closestPointCost(x_ptr, y_ptr, xdims_ptr, ydims_ptr, res_ptr, N);
}

void pyScalePoints(py::object x, py::object xdims, float scale, float t0, float t1, float t2, int N){
  float** x_ptr = getGPUPointer<float*>(x);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  scalePoints(x_ptr, xdims_ptr, scale, t0, t1, t2, N);
}

void pyInitProbNM(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object xdims, py::object ydims, int N,
		  float outlierprior, float outlierfrac, float T, 
		  py::object corr_cm, py::object corr_rm){
  /*
   * Initilialized correspondence matrix returned in corr
   */
  float** x_ptr  = getGPUPointer<float*>(x);
  float** xw_ptr = getGPUPointer<float*>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float** y_ptr  = getGPUPointer<float*>(y);
  float** yw_ptr = getGPUPointer<float*>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);

  initProbNM(x_ptr, y_ptr, xw_ptr, yw_ptr, N, xdims_ptr, ydims_ptr, 
	     outlierprior, outlierfrac, T, corr_ptr_cm, corr_ptr_rm);
}

void pyNormProbNM(py::object corr_cm, py::object corr_rm, py::object xdims,
		  py::object ydims, int N, float outlier_frac, int norm_iters,
		  py::object row_coeffs, py::object rn_col_coeffs, py::object cn_col_coeffs){

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);
 
  float** r_c      = getGPUPointer<float*>(row_coeffs);
  float** rn_c_c   = getGPUPointer<float*>(rn_col_coeffs);
  float** cn_c_c   = getGPUPointer<float*>(cn_col_coeffs);

  int* xdims_ptr  = getGPUPointer<int>(xdims);
  int* ydims_ptr  = getGPUPointer<int>(ydims);

  normProbNM(corr_ptr_cm, corr_ptr_rm, xdims_ptr, ydims_ptr, 
	     N, outlier_frac, norm_iters, r_c, rn_c_c, cn_c_c);
}

void pyGetTargPts(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object corr_cm, py::object corr_rm ,
		  py::object row_coeffs, py::object rn_col_coeffs, py::object cn_col_coeffs,
		  py::object xdims, py::object ydims, float cutoff,
		  int N, py::object xt, py::object yt){
  /*
   * target vectors returned in xt and yt
   */

  float** x_ptr  = getGPUPointer<float*>(x);
  float** xw_ptr = getGPUPointer<float*>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float** y_ptr  = getGPUPointer<float*>(y);
  float** yw_ptr = getGPUPointer<float*>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);
 
  float** r_c      = getGPUPointer<float*>(row_coeffs);
  float** rn_c_c   = getGPUPointer<float*>(rn_col_coeffs);
  float** cn_c_c   = getGPUPointer<float*>(cn_col_coeffs);


  float** xt_ptr = getGPUPointer<float*>(xt);
  float** yt_ptr = getGPUPointer<float*>(yt);

  getTargPts(x_ptr, y_ptr, xw_ptr, yw_ptr, corr_ptr_cm, corr_ptr_rm, r_c, rn_c_c, cn_c_c,
	     xdims_ptr, ydims_ptr, cutoff, N, xt_ptr, yt_ptr);
}

void pyCheckCudaErr(){
  checkCudaErr();
}

void pyResetDevice(){
  resetDevice();
}

BOOST_PYTHON_MODULE(cuda_funcs) {
  py::def("float_gpu_print_arr", &pyFloatPrintArr, (py::arg("x"), py::arg("N")));
  py::def("int_gpu_print_arr", &pyIntPrintArr, (py::arg("x"), py::arg("N")));

  py::def("fill_mat", &pyFillMat, (py::arg("dest"), py::arg("vals"), py::arg("dims"), py::arg("N")));

  py::def("sq_diffs", &pySqDiffMat, (py::arg("x"), py::arg("y"), py::arg("z"), py::arg("N"), py::arg("overwrite")));

  py::def("closest_point_cost", &pyClosestPointCost, (py::arg("x"), py::arg("y"), py::arg("xdims"), py::arg("ydims"), py::arg("N")));

  py::def("scale_points", &pyScalePoints, (py::arg("x"), py::arg("xdims"), py::arg("scale"), py::arg("t0"), py::arg("t1"), py::arg("t2"), py::arg("N")));

  py::def("init_prob_nm", &pyInitProbNM, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("xdims"), py::arg("ydims"), py::arg("N"), 
					  py::arg("outlierprior"), py::arg("outlierfrac"), py::arg("T"), 
					  py::arg("corr_cm"), py::arg("corr_rm")));

  py::def("norm_prob_nm", &pyNormProbNM, (py::arg("corr_cm"), py::arg("corr_rm"),
					  py::arg("xdims"), py::arg("ydims"), 
					  py::arg("N"), py::arg("outlier_frac"), 
					  py::arg("norm_iters"), py::arg("row_coeffs"),
					  py::arg("rn_col_coeffs"), py::arg("cn_col_coeffs")));

  py::def("get_targ_pts", &pyGetTargPts, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("corr_cm"), py::arg("corr_rm"), 
					  py::arg("row_coeffs"), py::arg("rn_col_coeffs"), 
					  py::arg("cn_col_coeffs"),
					  py::arg("xdims"), py::arg("ydims"), 
					  py::arg("outlier_cutoff"), py::arg("N"), 
					  py::arg("xt"), py::arg("yt")));
  py::def("check_cuda_err", &pyCheckCudaErr);
  py::def("reset_cuda", &pyResetDevice);
}
