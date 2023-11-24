cmake_minimum_required(VERSION 3.0.0)

project(transposeCUDA VERSION 1.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_library(transposeLib STATIC src/transpose.cu
            headers/CudaFuncs.hpp
            headers/ThreadPool.hpp
            headers/WorkerInterface.hpp
            headers/Matrix.hpp)
target_include_directories(transposeLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/headers)
set_target_properties(transposeLib PROPERTIES CUDA_ARCHITECTURES "35;50;72") # If you have different cuda arch you should add yours to the string separated by ;

add_executable(transposeCUDA src/main.cpp) # main used as usage show case and testing playground 
set_target_properties(transposeCUDA PROPERTIES CUDA_ARCHITECTURES "35;50;72") # If you have different cuda arch you should add yours to the string separated by ;
target_link_libraries(transposeCUDA PUBLIC transposeLib)