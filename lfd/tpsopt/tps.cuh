#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_DIM 150
#define DATA_DIM 3

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

void gpuPrintArr(float* x, int N);
void gpuPrintArr(int* x, int N);

void fillMat(float* dest_ptr[], float* val_ptr[], int* dims, int N);

void sqDiffMat(float* x_ptr[], float* y_ptr[], float* z, int N, bool overwrite);

void gramMatDist(float* x_ptr[], float* y_ptr[], int* dims, float sigma, float* z, int N);

void closestPointCost(float* x_ptr[], float* y_ptr[], int* xdims, int* ydims, float* res, int N);

void scalePoints(float* x_ptr[], int* xdims, float scale, float t0, float t1, float t2, int N);

void initProbNM(float* x[], float* y[], float* xw[], float* yw[],
		int N, int* xdims, int* ydims, float outlierprior, float outlierfrac,
		float T, float* corr_cm[], float* corr_rm[]);

void normProbNM(float* corr_cm[], float* corr_rm[], int* xdims, int* ydims, int N, 
		float outlier_frac, int norm_iters,
		float* row_c_res[], float* cm_col_c_res[], float* rm_col_c_res[]);

void getTargPts(float* x[], float* y[], float* xw[], float* yw[], 
		float* corr_cm[], float* corr_rm[], 
		float* row_c_ptrs[], float* cm_col_c_ptrs[], float* rm_col_c_ptrs[],
		int* xdims, int* ydims, float cutoff, int N,
		float* xt[], float* yt[]);

void checkCudaErr();

void resetDevice();
