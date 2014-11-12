cmake_minimum_required(VERSION 2.8)
project(lfd)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# http://cmake.3232098.n2.nabble.com/Default-value-for-CMAKE-BUILD-TYPE-td7550756.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BUILD_SHARED_LIBS true)

# external libraries
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/boost-python.cmake")

add_subdirectory(src)
