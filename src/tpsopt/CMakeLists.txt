# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

## template for cuda -> boost python
cuda_add_library(tps SHARED tps.cu)
target_link_libraries(tps ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

boost_python_module(cuda_funcs cuda_funcs.cpp)
target_link_libraries(cuda_funcs tps)
