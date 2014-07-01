# Append current NVCC flags by something, eg comput capability
cmake_minimum_required(VERSION 2.8)

project(tps-opt)
include(boost-python.cmake)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda-6.0/include)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

set(CMAKE_CXX_FLAGS "-fPIC")

set(CMAKE_BUILD_TYPE RELEASE)

# CUDA_ADD_LIBRARY(cdist_lib cdist.cu)

include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
# target_link_libraries(cdist_lib /usr/local/cuda/lib64/libcudart.so)
# add_executable(test_boost test_boost.cpp)
# boost_python_module(test_boost test_boost.cpp)

# target_link_libraries(test_boost cdist_lib)

# CUDA_ADD_EXECUTABLE(tps tps.cu)

## template for cuda -> boost python
CUDA_ADD_LIBRARY(tps SHARED tps.cu)
target_link_libraries(tps /usr/local/cuda/lib64/libcudart.so)
## necessary for cuBLAS
target_link_libraries(tps /usr/local/cuda/lib64/libcublas.so)
boost_python_module(cuda_funcs cuda_funcs.cpp)
target_link_libraries(cuda_funcs tps)





# boost_python_module(../bin/test_boost test_boost.cpp)

# Some untested, alternative statements of how to add libraries to an CUDA executable

# 
# 
# target_link_libraries(gpuSquareDemo externals)
