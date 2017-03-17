cmake_minimum_required(VERSION 2.8)
project(gipuma)

##################################
# Modified CMakeLists for gipuma #
##################################

find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED)

set(gipuma_lib_files
        config.h
        exception.h
        gipuma.h
        helper_cuda.h
        helper_string.h
        imageinfo.h
        managed.h
        mathUtils.h
        cameraGeometryUtils.h
        vector_operations.h
        camera.h
        globalstate.h
        algorithmparameters.h
        cameraparameters.h
        linestate.h
        groundTruthUtils.h
        displayUtils.h
        fileIoUtils.h
        gipuma.cu
        main.h
        ModifiedGipumaAPI.hpp
        )

set(gipuma_demo_files
        main.cpp
        )

if (CUDA_FOUND)

    #set(GIPUMA_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Absolute path of gipuma")
    set(GIPUMA_INCLUDE_DIR
            ${OpenCV_INCLUDE_DIRS}
            ${GIPUMA_PATH}
            ${CUDA_TOOLKIT_INCLUDE}
            CACHE PATH "include dirs for gipuma")
    include_directories(${GIPUMA_INCLUDE_DIR})

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3  --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_30,code=sm_30)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_52,code=sm_52)

    if (CMAKE_COMPILER_IS_GNUCXX)
        add_definitions(-std=c++11)
        add_definitions(-Wall)
        add_definitions(-Wextra)
        add_definitions(-pedantic)
        add_definitions(-Wno-unused-function)
        add_definitions(-Wno-switch)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
    endif ()

    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif ()


    # For compilation ...
    # Specify target & source files to compile it from
    #cuda_add_executable(
    cuda_add_library(
            gipuma
            ${gipuma_lib_files}
    )

    #target_include_directories(gipuma )

    add_executable(gipumaDemo
            ${gipuma_demo_files}
            )

    # For linking ...
    # Specify target & libraries to link it with
    target_link_libraries(
            gipumaDemo
            gipuma
            ${OpenCV_LIBS}
    )

    # link created library with dependencies too
    target_link_libraries(
            gipuma
            ${OpenCV_LIBS}
    )



    set(GIPUMA_FOUND TRUE CACHE BOOL "gipuma implementation found")
else (CUDA_FOUND)
    set(GIPUMA_FOUND FALSE CACHE BOOL "gipuma implementation not present")
    message(INFO "Error with gipuma library")
endif (CUDA_FOUND)
