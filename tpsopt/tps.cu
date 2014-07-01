#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tps.cuh"

/**************************************************************
 ****************Device Utility Functions**********************
 **************************************************************/

__device__ int rMInd(int offset, int i, int j, int n_cols){
  /*
   * returns an index into an array with n_cols colums
   * at row i, column j (A[i, j]) stored in Row-Major Format
   */
  return offset + i * n_cols + j;
}
__device__ int rMInd(int i, int j, int n_cols){
  /*
   * returns an index into an array with n_cols colums
   * at row i, column j (A[i, j]) stored in Row-Major Format
   */
  return i * n_cols + j;
}

__device__ int cMInd(int offset, int i, int j, int n_rows){
  /*
   * returns an index into an array with n_rows rows
   * at row i, column j (A[i, j]) stored in Column-Major Format
   */
  return offset + i + n_rows * j;
}

__device__ int cMInd(int i, int j, int n_rows){
  /*
   * returns an index into an array with n_rows rows
   * at row i, column j (A[i, j]) stored in Column-Major Format
   */
  return i + n_rows * j;
}

/************************************************************
 *********************GPU Kernels****************************
 ************************************************************/
__global__ void _gpuFloatPrintArr(float* x, int N){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N){
    printf("GPU Print:\t arr[%i] = %f\n", ix, x[ix]);
  }
}

__global__ void _gpuIntPrintArr(int* x, int N){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N){
    printf("GPU Print:\t arr[%i] = %i\n", ix, x[ix]);
  }
}

__global__ void _fillMat(float* dest_ptr[], float* val_ptr[], int* dims){
  /*
   * Fills the matrices pointed to in dest with the colum values in vals
   * called with 1 block per item and at least dims[bix] threads
   */
  __shared__ int dim;
  __shared__ float s_val[DATA_DIM], *dest;
  int bix = blockIdx.x; int tix = threadIdx.x;
  if (tix == 0) {
    dim = dims[bix];
    dest = dest_ptr[bix];
    float* val = val_ptr[bix];
    for(int i = 0; i < DATA_DIM; ++i){
      s_val[i] = val[i];
    }
  }
  __syncthreads();
  
  if (tix < dim){
    for (int i = 0; i < DATA_DIM; ++i){
      dest[rMInd(tix, i, DATA_DIM)] = s_val[i];
    }
  }    
}

__global__ void _sqDiffMat(float* x_ptr[], float* y_ptr[], float* z, bool overwrite){
  /*
   * called with 1 block per item comparison, 1 thread per row
   * computes the pointwise sum of squared differences for 
   * the vectors pointed to by x and y
   * the result is added to the float pointed to by res[bix]
   */
  int tix = threadIdx.x; int bix = blockIdx.x; int zix = rMInd(bix, tix, MAX_DIM);
  float *x, *y, diff, sum;
  x = x_ptr[bix]; y = y_ptr[bix];
  sum = 0;
  int ind = rMInd(tix, 0, DATA_DIM);  
  for (int i = 0; i < DATA_DIM; ++i){
    diff = x[ind + i] - y[ind + i];
    sum += diff * diff;
  }
  if (overwrite){
    z[zix] = sum;
  }
  else {
    z[zix] += sum;
  }
}


__global__ void _corrReduce(float* d1_ptr[], float* d2_ptr[], float* out_ptr[], float T){
  /* Takes pointers to two arrays of pairwise distances b/t forward and backward
   * warps. Puts the result in out_ptr
   * Called with 1 block/ptr and MAX_DIM threads
   */
  int tix = threadIdx.x; int bix = blockIdx.x;
  int ind;
  float *d1, *d2, *out;
  d1  = d1_ptr[bix];
  d2  = d2_ptr[bix];
  out = out_ptr[bix];
  for (int i = 0; i < MAX_DIM; ++i){
    ind = rMInd(i, tix, MAX_DIM);
    out[ind] = exp( ( -1 * sqrt(d1[ind]) - sqrt(d2[ind])) / ( (float) 2 * T));
  }
}

__global__ void _initProbNM(float* x_ptr[], float* y_ptr[], float* xw_ptr[], float* yw_ptr[], 
			    int* xdims, int* ydims, float outlierprior, float outlierfrac, float T, 
			    float* corr_ptr_cm[], float* corr_ptr_rm[]) {
  /* Batch Initialize the correspondence matrix for use in TPS-RPM
   * Called with 1 Block per item in the batch and MAX_DIM threads
   * assumes data is padded with 0's beyond bounds
   * Timing -- compute constrained
   *     -  Normal                      1.5440ms
   *     -  No P write                  1.6959ms
   *     -  Minimal Mem                 1.4142ms
   *     -  Single exp, contig writes   829.90us
   *     -  No Shared Mem               1.6208ms
   */
  __shared__ float s_x[MAX_DIM * DATA_DIM], s_y[MAX_DIM * DATA_DIM];
  __shared__ float s_xw[MAX_DIM * DATA_DIM], s_yw[MAX_DIM * DATA_DIM];
  __shared__ int xdim, ydim, m_dim;
  __shared__ float *x, *y, *xw, *yw, *corr_rm, *corr_cm;
  int tix = threadIdx.x; int bix = blockIdx.x;
  float dist_ij, dist_ji, tmp, diff; int i_ix, j_ix, n_corr_c, n_corr_r;
  if (tix == 0) {
    xdim = xdims[bix];
    ydim = ydims[bix];

    x = x_ptr[bix];
    y = y_ptr[bix];

    xw = xw_ptr[bix];
    yw = yw_ptr[bix];

    corr_cm = corr_ptr_cm[bix];
    corr_rm = corr_ptr_rm[bix];
  }
  __syncthreads();
  n_corr_r = xdim + 1;
  n_corr_c = ydim + 1;
  if (tix < MAX_DIM){
    for (int i = 0; i < DATA_DIM; ++i){
      s_x[rMInd(tix, i, DATA_DIM)]  = x[rMInd(tix, i, DATA_DIM)];
      s_xw[rMInd(tix, i, DATA_DIM)] = xw[rMInd(tix, i, DATA_DIM)];
    }
  }
  if (tix < MAX_DIM){
    for (int i = 0; i < DATA_DIM; ++i){
      s_y[rMInd(tix, i, DATA_DIM)]  = y[rMInd(tix, i, DATA_DIM)];
      s_yw[rMInd(tix, i, DATA_DIM)] = yw[rMInd(tix, i, DATA_DIM)];
    }
  }
  //Initialize the bottom right
  if (tix == 0){
    corr_rm[rMInd(xdim, ydim, n_corr_c)] = outlierfrac * sqrt((float) (xdim * ydim));
    corr_cm[cMInd(xdim, ydim, n_corr_r)] = outlierfrac * sqrt((float) (xdim * ydim));
    m_dim = MAX(xdim, ydim);
  }
  __syncthreads();

  i_ix = rMInd(tix, 0, DATA_DIM);
  for( int j = 0; j < m_dim; ++j){      
    j_ix = rMInd(j, 0, DATA_DIM);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff= s_xw[i_ix + k] - s_y[j_ix + k];
      tmp += diff * diff;
    }
    dist_ij = sqrt(tmp);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff= s_yw[i_ix + k] - s_x[j_ix + k];
      tmp += diff * diff;
    }
    dist_ji = sqrt(tmp);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff = s_x[i_ix + k] - s_yw[j_ix + k];
      tmp += diff * diff;
    }
    dist_ij += sqrt(tmp);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff = s_y[i_ix + k] - s_xw[j_ix + k];
      tmp += diff * diff;
    }
    dist_ji += sqrt(tmp);

    if (tix < xdim) corr_cm[cMInd(tix, j, n_corr_r)] = exp( -1 * dist_ij / (float) (2 * T)) + 1e-9;
    if (tix < ydim) corr_rm[rMInd(j, tix, n_corr_c)] = exp( -1 * dist_ji / (float) (2 * T)) + 1e-9;
  }
  if (tix < xdim) {
    corr_cm[cMInd(tix, ydim, n_corr_r)] = outlierprior;
    corr_rm[rMInd(tix, ydim, n_corr_c)] = outlierprior;
  }
  if (tix < ydim) {
    corr_cm[cMInd(xdim, tix, n_corr_r)] = outlierprior;
    corr_rm[rMInd(xdim, tix, n_corr_c)] = outlierprior;
  }
}


__global__ void _normProbNM(float* corr_ptr_cm[], float* corr_ptr_rm[], int* xdims, int* ydims,
			    int N, float outlierfrac, int norm_iters, 
			    float* row_c_res[], float* cm_col_c_res[], float* rm_col_c_res[]){
  /*  row - column normalizes prob_nm
   *  Launch with 1 block per matrix, store xdims, ydims, stride, N in constant memory
   *  Thread.idx governs which row/column to normalize
   *  Assumed to have more than 7 threads
   *  ---Might be able to run without synchronization
   *  1. Set up shared memory
   *  2. Sums rows
   *  3. Norm rows
   *  4. Sum Columns
   *  5. Norm Colums -- repeat
   * Timing -- memory limited
   *    -  Minimal Memory Accesses:       407.97us
   *    -  Write (only) to corr_rm/cm:    913.71us
   *    -  Read  (only) from corr_rm/cm:  1.9716ms
   *    -  RW from corr_rm/cm:            2.6585ms
   */
  //set up shared variables to be read once
  __shared__ int n_corr_r, n_corr_c;
  __shared__ float *corr_rm, *corr_cm, *row_c, *cm_col_c, *rm_col_c;
  __shared__ float col_coeffs[MAX_DIM], row_coeffs[MAX_DIM];
  float r_sum, c_sum, r_tgt, c_tgt;
  int bix = blockIdx.x; int tix = threadIdx.x; int ind;
  if (tix == 0) { 
    n_corr_r = xdims[bix] + 1;
    n_corr_c = ydims[bix] + 1;
    corr_cm = corr_ptr_cm[bix];
    corr_rm = corr_ptr_rm[bix];

    row_c    = row_c_res[bix];
    cm_col_c = cm_col_c_res[bix];
    rm_col_c = rm_col_c_res[bix];
  }
  row_coeffs[tix] = 1;
  col_coeffs[tix] = 1;
  __syncthreads();
  if (tix == n_corr_r-1){
    r_tgt = ((float) (n_corr_c-1)) * outlierfrac;
  } else{
    r_tgt = 1;
  }
  if (tix == n_corr_c-1){
    c_tgt = ((float) (n_corr_r-1)) * outlierfrac;
  } else{
    c_tgt = 1;
  }
  c_sum = c_tgt;
  //do normalization
  for(int ctr = 0; ctr < norm_iters; ++ctr){
    if (tix < n_corr_c){
      //sum cols and divide      
      col_coeffs[tix] = c_tgt / c_sum;
    }
    __syncthreads();
    r_sum = 0;
    if (tix < n_corr_r){
      //sum rows and divide
      for (int i = 0; i < n_corr_c; ++i) {
  	r_sum = r_sum + corr_cm[cMInd(tix, i, n_corr_r)] * col_coeffs[i];
      }
      row_coeffs[tix] = r_tgt / r_sum;
      if (ctr == norm_iters - 1){
	//write back the sums into the final column
	ind = cMInd(tix, n_corr_c-1, n_corr_r);
	//just subtract off the last value
	//1 - (col_coeffs[-1] * 1/row_sum * corr_cm[tix, -1])
	corr_cm[ind] = 1 - col_coeffs[n_corr_c-1] * corr_cm[ind] * row_coeffs[tix];
      }
    }
    __syncthreads();
    if (tix < n_corr_c){
      c_sum = 0;
      for (int i = 0; i < n_corr_r; ++i) {
  	c_sum = c_sum + corr_rm[rMInd(i, tix, n_corr_c)] * row_coeffs[i];
      }
      if (ctr == norm_iters - 1){
	ind = rMInd(n_corr_r - 1, tix, n_corr_c);
	corr_rm[ind] = 1 - corr_rm[ind] * row_coeffs[n_corr_r - 1] * c_tgt / c_sum;
      }
    }
  }
  //copy results back
  //corr_cm row-normalized
  //corr_rm column-normalized
  //row coefficients stored in corr[MAX_DIM, :]
  //column coefficients stored in corr[:, MAX_DIM]
  //row_coeffs * col_coeffs is row normalizer
  //row_coeffs * col_sums is column normalizer
  row_c[tix]    = row_coeffs[tix];
  cm_col_c[tix] = col_coeffs[tix];
  rm_col_c[tix] = c_tgt/c_sum;
}

__global__ void  _getTargPts(float* x_ptr[], float* y_ptr[], float* xw_ptr[], float*yw_ptr[],
			     float* corr_ptr_cm[], float* corr_ptr_rm[],
			     float* row_c_ptrs[], float* cm_col_c_ptrs[], float* rm_col_c_ptrs[],
			     int* xdims, int* ydims, float cutoff,
			     int N, float* xt_ptr[], float* yt_ptr[]){
  /*  Computes the target points for x and y when warped
   *  Launch with 1 block per item
   *  Thread.idx governs which row/column we are dealing with
   *  Assumed to have more than 4 threads
   *  
   *  1. set up shared memory
   *  2. Norm rows of corr, detect source outliers
   *  3. Update xt with correct value (0 pad other areas
   *  4. Norm cols of corr, detect target outliers
   *  5. Update yt with correct value (0 pad other areas
   *  Timing -- memory/compute limited
   *    -  Minimal Memory Accesses:       875.12us
   *    -  Only Reading x/y into s:       748.21us
   *    -  Reading the sum:               711.14us (Why is this so much faster)
   *    -  Reads from xw/yw:              613.57us
   *    -  No sum/everything else:        838.12us
   *    -  Full:                          1.6679ms                  
   */
  __shared__ int xdim, ydim; int n_corr_r, n_corr_c;
  __shared__ float s_y[MAX_DIM * DATA_DIM], s_x[MAX_DIM * DATA_DIM];
  __shared__ float *x, *y, *xw, *yw, *xt, *yt, *corr_rm, *corr_cm, *row_c, *cm_col_c, *rm_col_c;
  int tix = threadIdx.x; int bix = blockIdx.x;  
  float targ0, targ1, targ2;
  if (threadIdx.x == 0){
    xdim     = xdims[bix];
    ydim     = ydims[bix];
    x        = x_ptr[bix];
    y        = y_ptr[bix];
    xw       = xw_ptr[bix];
    yw       = yw_ptr[bix];
    xt       = xt_ptr[bix];
    yt       = yt_ptr[bix];
    corr_cm  = corr_ptr_cm[bix];
    corr_rm  = corr_ptr_rm[bix];
    row_c    = row_c_ptrs[bix];
    cm_col_c = cm_col_c_ptrs[bix];
    rm_col_c = rm_col_c_ptrs[bix];
  }
  __syncthreads();  
  n_corr_r = xdim + 1; n_corr_c = ydim + 1;

  if (tix < xdim){
    for(int i = 0; i < DATA_DIM; ++i){
      s_x[rMInd(tix, i, DATA_DIM)] = x[rMInd(tix, i, DATA_DIM)];
    }
  }
  if (tix < ydim){
    for(int i = 0; i < DATA_DIM; ++i){
      s_y[rMInd(tix, i, DATA_DIM)] = y[rMInd(tix, i, DATA_DIM)];
    }
  }
  __syncthreads();

  if (tix < xdim){
    //if the point is an outlier map it to its current warp
    if (corr_cm[cMInd(tix, n_corr_c-1, n_corr_r)] < cutoff){      
      for(int i = 0; i < DATA_DIM; ++i){	
    	xt[rMInd(tix, i, DATA_DIM)] = xw[rMInd(tix, i, DATA_DIM)];
      }
    } else {
      targ0 = 0; targ1 = 0; targ2 = 0;
      for(int j = 0; j < ydim; ++j){
	targ0 = targ0 + corr_cm[cMInd(tix, j, n_corr_r)] * cm_col_c[j]
	  * s_y[rMInd(j, 0, DATA_DIM)] ;
	targ1 = targ1 + corr_cm[cMInd(tix, j, n_corr_r)] * cm_col_c[j]
	  * s_y[rMInd(j, 1, DATA_DIM)] ;
	targ2 = targ2 + corr_cm[cMInd(tix, j, n_corr_r)] * cm_col_c[j]
	  * s_y[rMInd(j, 2, DATA_DIM)] ;
      }
      xt[rMInd(tix, 0, DATA_DIM)] = targ0 * row_c[tix];
      xt[rMInd(tix, 1, DATA_DIM)] = targ1 * row_c[tix];
      xt[rMInd(tix, 2, DATA_DIM)] = targ2 * row_c[tix];
    }
  }
  if (tix < ydim){
    if (corr_rm[rMInd(n_corr_r-1, tix, n_corr_c)] < cutoff){
      for(int i = 0; i < DATA_DIM; ++i){
  	yt[rMInd(tix, i, DATA_DIM)] = yw[rMInd(tix, i, DATA_DIM)];
      }
    } else {
      targ0 = 0; targ1 = 0; targ2 = 0;
      for(int j = 0; j < xdim; ++j){
	targ0 = targ0 + corr_rm[rMInd(j, tix, n_corr_c)] * row_c[j]
	  * s_x[rMInd(j, 0, DATA_DIM)];
	targ1 = targ1 + corr_rm[rMInd(j, tix, n_corr_c)] * row_c[j]
	  * s_x[rMInd(j, 1, DATA_DIM)];
	targ2 = targ2 + corr_rm[rMInd(j, tix, n_corr_c)] * row_c[j]
	  * s_x[rMInd(j, 2, DATA_DIM)];
      }
      yt[rMInd(tix, 0, DATA_DIM)] = targ0 * rm_col_c[tix];
      yt[rMInd(tix, 1, DATA_DIM)] = targ1 * rm_col_c[tix];
      yt[rMInd(tix, 2, DATA_DIM)] = targ2 * rm_col_c[tix];
    }
  } else if (tix < MAX_DIM){
    for(int i = 0; i < DATA_DIM; ++i){
      yt[rMInd(tix, i, DATA_DIM)] = 0;
    }
  }
}


/*****************************************************************************
 *******************************Wrappers**************************************
 *****************************************************************************/

void gpuPrintArr(float* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuFloatPrintArr<<<n_blocks, n_threads>>>(x, N);
  cudaDeviceSynchronize();
}

void gpuPrintArr(int* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuIntPrintArr<<<n_blocks, n_threads>>>(x, N);
  cudaDeviceSynchronize();
}

void fillMat(float* dest_ptr[], float* val_ptr[], int* dims, int N){
  int n_threads = MAX_DIM;
  int n_blocks = N;
  _fillMat<<<n_blocks, n_threads>>>(dest_ptr, val_ptr, dims);
}

void sqDiffMat(float* x_ptr[], float* y_ptr[], float* z, int N, bool overwrite){
  _sqDiffMat<<<N, MAX_DIM>>>(x_ptr, y_ptr, z, overwrite);
}

void initProbNM(float* x[], float* y[], float* xw[], float* yw[],
		int N, int* xdims, int* ydims, float outlierprior, float outlierfrac,
		float T, float* corr_cm[], float* corr_rm[]){
  int n_threads = MAX_DIM;
  int n_blocks = N;
  // printf("Launching Initlization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _initProbNM<<<n_blocks, n_threads>>>(x, y, xw, yw, xdims, ydims, outlierprior, outlierfrac, T,
				       corr_cm, corr_rm);
}

void normProbNM(float* corr_cm[], float* corr_rm[], int* xdims, int* ydims, int N, 
		float outlier_frac, int norm_iters,
		float* row_c_res[], float* cm_col_c_res[], float* rm_col_c_res[]){
  int n_blocks = N;
  int n_threads = MAX_DIM;
  // printf("Launching Normalization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _normProbNM<<<n_blocks, n_threads>>>(corr_cm, corr_rm, xdims, ydims, 
				       N, outlier_frac, norm_iters,
				       row_c_res, cm_col_c_res, rm_col_c_res);
}


void getTargPts(float* x[], float* y[], float* xw[], float* yw[], 
		float* corr_cm[], float* corr_rm[], 
		float* row_c_ptrs[], float* cm_col_c_ptrs[], float* rm_col_c_ptrs[],
		int* xdims, int* ydims, float cutoff, int N,
		float* xt[], float* yt[]){
  int n_blocks = N;
  int n_threads = MAX_DIM;
  // printf("Launching Get Targ Pts Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _getTargPts<<<n_blocks, n_threads>>>(x, y, xw, yw, corr_cm, corr_rm, 
				       row_c_ptrs, cm_col_c_ptrs, rm_col_c_ptrs, xdims, ydims, 
				       cutoff, N, xt, yt);
}

void checkCudaErr(){
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("Error Detected:\t%s\n", cudaGetErrorString(err));
    cudaDeviceReset();
    exit(1);
  }
}
void resetDevice(){
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("Error Detected:\t%s\n", cudaGetErrorString(err));
    cudaDeviceReset();
    exit(1);
  }
  cudaDeviceReset();
}
