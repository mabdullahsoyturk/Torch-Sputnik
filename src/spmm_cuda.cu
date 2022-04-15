#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)

torch::Tensor spmm(int m, int k, int n, int nonzeros,
               torch::Tensor row_indices, torch::Tensor values,
               torch::Tensor row_offsets, torch::Tensor column_indices,
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();
    float* _values = values.data_ptr<float>();
    int* _row_indices = row_indices.data_ptr<int>();
    int* _row_offsets = row_offsets.data_ptr<int>();
    int* _column_indices = column_indices.data_ptr<int>();
    float* _dense_matrix = dense_matrix.data_ptr<float>();
    float* _output_matrix = output_matrix.data_ptr<float>();

    CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, 
                                _row_indices, 
                                _values,
                                _row_offsets, 
                                _column_indices,
                                _dense_matrix, 
                                _output_matrix, 
                                stream));
    cudaDeviceSynchronize();
    
    return output_matrix;
}
