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
    
    CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzeros, 
                                row_indices.data_ptr<int>(), 
                                row_offsets.data_ptr<int>(), 
                                column_indices.data_ptr<int>(),
                                lhs_matrix.data_ptr<float>(), 
                                rhs_matrix.data_ptr<float>(), 
                                output_values.data_ptr<float>(), 
                                stream));
    cudaDeviceSynchronize();
    
    return output_values;
}
