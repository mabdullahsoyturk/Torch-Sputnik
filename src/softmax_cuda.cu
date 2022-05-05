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

torch::Tensor softmax(int m, int n, int nonzeros,
                      torch::Tensor values,
                      torch::Tensor row_indices,
                      torch::Tensor row_offsets,
                      torch::Tensor column_indices) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);
    torch::Tensor out = torch::zeros({m, n}, options);

    CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzeros, 
                                values.data_ptr<float>(),
                                row_indices.data_ptr<int>(), 
                                row_offsets.data_ptr<int>(), 
                                column_indices.data_ptr<int>(),
                                out.data_ptr<float>(), 
                                stream));
    cudaDeviceSynchronize();
    
    return out;
}
