#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sddmm(int m, int k, int n, torch::Tensor nnzs,
                           torch::Tensor row_indices,
                           torch::Tensor row_offsets,
                           torch::Tensor column_indices,
                           torch::Tensor lhs_matrix,
                           torch::Tensor rhs_matrix,
                           torch::Tensor mask) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int dim_offset = nnzs.size(0) - 1;
    int replication = dim_offset == 1 ? nnzs.size(0) : 1;

    int* nonzeros = nnzs.data_ptr<int>();
    int sum_nonzeros = 0;

    for(int i = 0; i < nnzs.size(0); i++) {
      sum_nonzeros += nonzeros[i];
    }

    int sum = 0;

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzeros[idx], 
                                row_indices.data_ptr<int>() + m * idx, 
                                row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                column_indices.data_ptr<int>() + sum,
                                lhs_matrix.data_ptr<float>() + m * k * idx, 
                                rhs_matrix.data_ptr<float>() + k * n * idx, 
                                mask.data_ptr<float>() + sum, 
                                stream));
      sum += nonzeros[idx];
    }
    
    cudaStreamSynchronize(stream);
    
    return mask;
}
