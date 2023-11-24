#include "CudaFuncs.hpp"

#define THREADS_PER_BLOCK 512
typedef int myType;

__global__ void transposeMatrix(myType * matrix_in, myType * matrix_out, const int width, const int height) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < width * height)
    {
        const int i = index / width;
        const int j = index - i * width;

        matrix_out[i + j * height] = matrix_in[j + i * width];
    }
}

Matrix cudaWrapper(Matrix & matrix, cudaStream_t stream) {
    const int width = matrix.width, height = matrix.height;
    if (matrix.width == 0 || matrix.height == 0) // Bad matrices are not processed
        return matrix;

    const int n = width * height;
    const int size = n * (sizeof(myType));

    // Allocating memory
    myType *host_result;
    cudaMallocHost((void **) &host_result, size);
    myType *d_matrix, *d_result;

    cudaMalloc((void **) &d_matrix, size);
    cudaMalloc((void **) &d_result, size);

    // H2D memory copy
    cudaMemcpyAsync(d_matrix, matrix.data.data(), size, cudaMemcpyHostToDevice, stream);

    transposeMatrix<<<(n / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 0, stream>>>(d_matrix, d_result, width, height);

    // D2H memory copy
    cudaMemcpyAsync(host_result, d_result, size, cudaMemcpyDeviceToHost, stream);

    std::swap(matrix.width, matrix.height);
    
    cudaStreamSynchronize(stream);

    // Saving results
    for (size_t i = 0; i < width*height; i++)
    {
        matrix.data[i] = host_result[i];
    }

    cudaFreeHost(host_result);

    cudaFreeAsync(d_matrix, stream);
    cudaFreeAsync(d_result, stream);

    return matrix;
}