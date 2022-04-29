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
               torch::Tensor row_indices, 
               torch::Tensor values,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix, 
               torch::Tensor bias) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);
    torch::Tensor out = torch::zeros({m, n}, options);
    
    CUDA_CALL(sputnik::CudaSpmmBiasRelu(m, k, n, nonzeros, 
                                row_indices.data_ptr<int>(), 
                                values.data_ptr<float>(),
                                row_offsets.data_ptr<int>(), 
                                column_indices.data_ptr<int>(),
                                dense_matrix.data_ptr<float>(),
                                bias.data_ptr<float>(), 
                                out.data_ptr<float>(), 
                                stream));
    cudaDeviceSynchronize();
    
    return out;
}
