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
    /*--- CHECKS ---*/
    assert(row_indices.dim() == 1); // Row indices should have 1 dimension
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    assert(row_indices.size(0) + 1 == row_offsets.size(0)); // Row offsets should have one more row than row indices
    assert(lhs_matrix.size(1) == rhs_matrix.size(1)); // Last dim of input matrices must match.
    assert(lhs_matrix.dim() == 2 || lhs_matrix.dim() == 3); // Expected 2-dim or 3-dim lhs matrix tensor
    assert(rhs_matrix.dim() == 2 || rhs_matrix.dim() == 3); // Expected 2-dim or 3-dim rhs matrix tensor
    assert(lhs_matrix.dim() == rhs_matrix.dim()); // Rhs and lhs must match number of dims
                            
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int nonzeros    = column_indices.size(0);
    int dim_offset  = lhs_matrix.dim() - 2;
    int replication = dim_offset == 1 ? lhs_matrix.size(0) : 1;
    
    int k           = lhs_matrix.size(dim_offset + 1);

    /*--- CHECKS ---*/
    assert(row_indices.size(0) == m); // Num row indices and 'm' must match
    assert(lhs_matrix.size(dim_offset) == m); // First dim of lhs must match output rows.
    assert(rhs_matrix.size(dim_offset) == n); // First dim of lhs must match output cols.
    assert(replication == 1 || replication == rhs_matrix.size(0)); // First dim of lhs & rhs must match

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, lhs_matrix.device().index());

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
    
    return output;
}
