#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include "error_check.h"

torch::Tensor sddmm_many_mask(int b, int m, int n,
                              torch::Tensor nonzeros,
                              torch::Tensor row_indices,
                              torch::Tensor row_offsets,
                              torch::Tensor column_indices,
                              torch::Tensor lhs_matrix,
                              torch::Tensor rhs_matrix) {
    /*--- CHECKS ---*/
    assert(row_indices.dim() == 1); // Row indices should have 1 dimension
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    assert(lhs_matrix.size(-1) == rhs_matrix.size(-1)); // Last dim of input matrices must match.
    assert(lhs_matrix.dim() == 2 || lhs_matrix.dim() == 3); // Expected 2-dim or 3-dim lhs matrix tensor
    assert(rhs_matrix.dim() == 2 || rhs_matrix.dim() == 3); // Expected 2-dim or 3-dim rhs matrix tensor
    assert(lhs_matrix.dim() == rhs_matrix.dim()); // Rhs and lhs must match number of dims
                            
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int dim_offset  = lhs_matrix.dim() - 2;
    int replication = dim_offset == 1 ? lhs_matrix.size(0) : 1;
    int k           = lhs_matrix.size(dim_offset + 1);
    int num_heads   = replication / b;

    int max_nonzeros = -1;

    for(int i = 0; i < nonzeros.size(0); i++) {
        if(nonzeros[i].item<int>() > max_nonzeros) {
            max_nonzeros = nonzeros[i].item<int>();
        }
    }

    /*--- CHECKS ---*/
    assert(lhs_matrix.size(dim_offset) == m); // First dim of lhs must match output rows.
    assert(rhs_matrix.size(dim_offset) == n); // First dim of lhs must match output cols.
    assert(replication == 1 || replication == rhs_matrix.size(0)); // First dim of lhs & rhs must match

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, lhs_matrix.device().index());

    float minus_inf = -std::numeric_limits<float>::infinity();
    torch::Tensor output = replication == 1 ? torch::full({max_nonzeros}, minus_inf, options) : torch::full({replication, max_nonzeros}, minus_inf, options);

    int column_indices_tracker = 0;
    for (int idx = 0; idx < replication; ++idx) {
        int batch_index = idx / num_heads;  
        int nonzero = nonzeros[batch_index].item<int>();

        if(idx != 0 && idx % num_heads == 0) {
            column_indices_tracker += nonzeros[batch_index - 1].item<int>();
        }

        CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzero, 
                                row_indices.data_ptr<int>() + m * batch_index, 
                                row_offsets.data_ptr<int>() + (m + 1) * batch_index, 
                                column_indices.data_ptr<int>() + column_indices_tracker,
                                lhs_matrix.data_ptr<float>() + m * k * idx, 
                                rhs_matrix.data_ptr<float>() + k * n * idx, 
                                output.data_ptr<float>() + max_nonzeros * idx, 
                                stream));
    }
    
    return output;
}
