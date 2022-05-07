#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor softmax(int m, int n, int nonzeros,
                      torch::Tensor values,
                      torch::Tensor row_indices,
                      torch::Tensor row_offsets,
                      torch::Tensor column_indices) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int dim_offset = values.dim() - 1;
    int replication = dim_offset == 1 ? values.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);
    torch::Tensor out = torch::zeros_like(values, options);

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzeros, 
                                values.data_ptr<float>() + nonzeros * idx,
                                row_indices.data_ptr<int>() + m * idx, 
                                row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                column_indices.data_ptr<int>() + nonzeros * idx,
                                out.data_ptr<float>() + nonzeros * idx, 
                                stream));      
    }

    cudaDeviceSynchronize();
    
    return out;
}
