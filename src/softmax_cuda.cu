#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sparse_softmax(torch::Tensor values,
                      torch::Tensor row_indices,
                      torch::Tensor row_offsets,
                      torch::Tensor column_indices) {
    /*--- CHECKS ---*/
    assert(values.dim() == 1 || values.dim() == 2); // Values should have 1 or 2 dimensions
    assert(row_indices.dim() == 1); // Row indices should have 1 dimension
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    assert(row_indices.size(0) + 1 == row_offsets.size(0)); // Row offsets should have one more row than row indices
    
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int m = row_indices.size(0);
    int n = -1; // The kernel doesn't actually need the n argument. Pass garbage.
    int nonzeros = column_indices.size(0);

    int dim_offset = values.dim() - 1;
    int replication = dim_offset == 1 ? values.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());
    
    torch::Tensor output = torch::zeros_like(values, options);

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzeros, 
                                values.data_ptr<float>() + nonzeros * idx,
                                row_indices.data_ptr<int>(), 
                                row_offsets.data_ptr<int>(), 
                                column_indices.data_ptr<int>(),
                                output.data_ptr<float>() + nonzeros * idx, 
                                stream));     
    }

    return output;
}
