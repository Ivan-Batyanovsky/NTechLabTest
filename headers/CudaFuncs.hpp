#ifndef __CUDA_FUNCTIONS_HPP__
#define __CUDA_FUNCTIONS_HPP__

#include <driver_types.h>
#include "Matrix.hpp"

 
/// @brief Wrapper around transposeMatrix which uses CUDA
/// @param matrix input matrix 
/// @param stream CUDA stream for async function calls 
/// @return Transposed matrix
Matrix cudaWrapper(Matrix &, cudaStream_t);

#endif