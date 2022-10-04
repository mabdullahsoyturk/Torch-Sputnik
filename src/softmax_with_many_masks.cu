#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sparse_softmax_many_mask(int b, int m,
                                        torch::Tensor nonzeros,
                                        torch::Tensor values,
                                        torch::Tensor row_indices,
                                        torch::Tensor row_offsets,
                                        torch::Tensor column_indices) {
    /*--- CHECKS ---*/
    assert(values.dim() == 1 || values.dim() == 2); // Values should have 1 or 2 dimensions
    assert(row_indices.dim() == 1); // Row indices should have 1 dimension
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int n = -1; // The kernel doesn't actually need the n argument. Pass garbage.

    int dim_offset = values.dim() - 1;
    int replication = dim_offset == 1 ? values.size(0) : 1;
    int num_heads = replication / b;

    int max_nonzeros = -1;

    for(int i = 0; i < nonzeros.size(0); i++) {
        if(nonzeros[i].item<int>() > max_nonzeros) {
            max_nonzeros = nonzeros[i].item<int>();
        }
    }

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());
    
    torch::Tensor output = torch::zeros_like(values, options);

    int column_indices_tracker = 0;
    for (int idx = 0; idx < replication; ++idx) {
        int batch_index = idx / num_heads;
        int nonzero = nonzeros[batch_index].item<int>();

        if(idx != 0 && idx % num_heads == 0) {
            column_indices_tracker += nonzeros[batch_index - 1].item<int>();
        }

        CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzero,
                                values.data_ptr<float>() + max_nonzeros * idx,
                                row_indices.data_ptr<int>() + m * batch_index, 
                                row_offsets.data_ptr<int>() + (m + 1) * batch_index, 
                                column_indices.data_ptr<int>() + column_indices_tracker,
                                output.data_ptr<float>() + max_nonzeros * idx, 
                                stream));     
    }

    return output;
}
