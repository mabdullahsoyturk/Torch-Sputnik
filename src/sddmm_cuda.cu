#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sddmm(int m, int n,
                           torch::Tensor row_indices,
                           torch::Tensor row_offsets,
                           torch::Tensor column_indices,
                           torch::Tensor lhs_matrix,
                           torch::Tensor rhs_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int nonzeros    = column_indices.size(-1);
    int dim_offset  = lhs_matrix.dim() - 2;
    int k           = lhs_matrix.size(dim_offset + 1);
    int replication = dim_offset == 1 ? lhs_matrix.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, lhs_matrix.device().index())
                                        .requires_grad(true);

    torch::Tensor output = replication == 1 ? torch::zeros({nonzeros}, options) : torch::zeros({replication, nonzeros}, options);

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzeros, 
                                row_indices.data_ptr<int>(), 
                                row_offsets.data_ptr<int>(), 
                                column_indices.data_ptr<int>(),
                                lhs_matrix.data_ptr<float>() + m * k * idx, 
                                rhs_matrix.data_ptr<float>() + k * n * idx, 
                                output.data_ptr<float>() + nonzeros * idx, 
                                stream));
    }
    
    cudaStreamSynchronize(stream);
    
    return output;
}
