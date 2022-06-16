#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sparse_softmax(torch::Tensor values,
                      torch::Tensor row_indices,
                      torch::Tensor row_offsets,
                      torch::Tensor column_indices) {
    CHECK_INPUT(values);
    CHECK_INPUT(row_indices);
    CHECK_INPUT(row_offsets);
    CHECK_INPUT(column_indices);
    
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int m = row_indices.size(-1);
    int n = -1; // The kernel doesn't actually need the n argument. Pass garbage.
    int nonzeros = column_indices.size(-1);

    int dim_offset = values.dim() - 1;
    int replication = dim_offset == 1 ? values.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index())
                                        .requires_grad(true);
    
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
