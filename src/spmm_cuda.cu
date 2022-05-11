#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor spmm(int m, int k, int n, torch::Tensor nnzs,
               torch::Tensor row_indices, 
               torch::Tensor values,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int dim_offset = dense_matrix.dim() - 2;
    int replication = dim_offset == 1 ? dense_matrix.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);

    torch::Tensor out = replication == 1 ? torch::zeros({m, n}, options) : torch::zeros({replication, m, n}, options);

    int* nonzeros = nnzs.data_ptr<int>();
    int sum = 0;
    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros[idx], 
                                  row_indices.data_ptr<int>() + m * idx, 
                                  values.data_ptr<float>() + sum,
                                  row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                  column_indices.data_ptr<int>() + sum,
                                  dense_matrix.data_ptr<float>() + k * n * idx,
                                  out.data_ptr<float>() + m * n * idx, 
                                  stream));
      sum += nonzeros[idx];
    }

    cudaStreamSynchronize(stream);
    
    return out;
}
