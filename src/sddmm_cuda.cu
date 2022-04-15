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

 torch::Tensor sddmm(int m, int k, int n, int nonzeros,
                           torch::Tensor row_indices,
                           torch::Tensor row_offsets,
                           torch::Tensor column_indices,
                           torch::Tensor lhs_matrix,
                           torch::Tensor rhs_matrix,
                           torch::Tensor output_values) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();
    
    int* _row_indices = row_indices.data_ptr<int>();
    int* _row_offsets = row_offsets.data_ptr<int>();
    int* _column_indices = column_indices.data_ptr<int>();
    float* _lhs_matrix = lhs_matrix.data_ptr<float>();
    float* _rhs_matrix = rhs_matrix.data_ptr<float>();
    float* _output_values = output_values.data_ptr<float>();

    CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzeros, 
                                _row_indices, 
                                _row_offsets, 
                                _column_indices,
                                _lhs_matrix, 
                                _rhs_matrix, 
                                _output_values, 
                                stream));
    cudaDeviceSynchronize();
    
    return output_values;
}
