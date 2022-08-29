#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

// Sparse (1D-2D) x Dense (2D-3D)
// Example: Sparse = 512x768, Dense = 4 * 768 * 1024, Result = 4x512x1024
torch::Tensor spmm(int m, int k,
               torch::Tensor values, 
               torch::Tensor row_indices,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense) {
    /*--- CHECKS ---*/
    assert(values.dim() == 1 || values.dim() == 2); // Values should have 1 or 2 dimensions
    assert(row_indices.dim() == 1); // Row indices should have 1 dimension
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    assert(dense.dim() == 2 || dense.dim() == 3); // Dense should have 2 or 3 dimensions
    assert(row_indices.size(0) + 1 == row_offsets.size(0)); // Row offsets should have one more row than row indices
    assert(values.dim() == dense.dim() - 1); // Values and dense must be replicated the same

    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int nonzeros = column_indices.size(-1);
   
    int dim_offset = dense.dim() - 2;
    int replication = dim_offset == 1 ? dense.size(0) : 1;
    
    int n = dense.size(dim_offset + 1);

    /*--- CHECKS ---*/
    // Validate the sparse matrix and dense matrix shapes match.
    assert(values.size(dim_offset) == nonzeros); // Num values must equal num column indices
    assert(row_indices.size(0) == m); // Num row indices and 'm' must match
    assert(dense.size(dim_offset) == k); // Inner matrix dimensions must match
    assert(replication == 1 || replication == values.size(0)); // First dim of values and dense must match

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());

    torch::Tensor out = replication == 1 ? torch::zeros({m, n}, options) : torch::zeros({replication, m, n}, options);

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, 
                                  row_indices.data_ptr<int>(), 
                                  values.data_ptr<float>() + nonzeros * idx,
                                  row_offsets.data_ptr<int>(), 
                                  column_indices.data_ptr<int>(),
                                  dense.data_ptr<float>() + k * n * idx,
                                  out.data_ptr<float>() + m * n * idx, 
                                  stream));
    }

    return out;
}
