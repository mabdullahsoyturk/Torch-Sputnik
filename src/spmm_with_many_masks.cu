#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

// Sparse (1D-2D) x Dense (2D-3D)
// Example: Sparse = 512x768, Dense = 4 * 768 * 1024, Result = 4x512x1024
torch::Tensor spmm_many_mask(int b, int m, int k,
                torch::Tensor nonzeros,    
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
    assert(values.dim() == dense.dim() - 1); // Values and dense must be replicated the same

    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int max_nonzeros = -1;

    for(int i = 0; i < nonzeros.size(0); i++) {
        if(nonzeros[i].item<int>() > max_nonzeros) {
            max_nonzeros = nonzeros[i].item<int>();
        }
    }
   
    int dim_offset  = dense.dim() - 2;
    int replication = dim_offset == 1 ? dense.size(0) : 1;
    int num_heads   = replication / b;
    
    int n = dense.size(dim_offset + 1);

    /*--- CHECKS ---*/
    // Validate the sparse matrix and dense matrix shapes match.
    std::cout << "Replication: " << replication << ", values.size(0): " << values.size(0) << std::endl;
    assert(replication == 1 || replication == values.size(0)); // First dim of values and dense must match

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());

    torch::Tensor output = replication == 1 ? torch::zeros({m, n}, options) : torch::zeros({replication, m, n}, options);

    int column_indices_tracker = 0;
    for (int idx = 0; idx < replication; ++idx) {
        int batch_index = idx / num_heads;  
        int nonzero = nonzeros[batch_index].item<int>();

        if(idx != 0 && idx % num_heads == 0) {
            column_indices_tracker += nonzeros[batch_index - 1].item<int>();
        }

        //std::cout << max_nonzeros << " " << nonzero << " " << column_indices_tracker << std::endl;

        //std::cout << "m: " << m << ", k: " << k << ", batch_index: " << batch_index << std::endl;

        CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzero, 
                                    row_indices.data_ptr<int>() + m * batch_index, 
                                    values.data_ptr<float>() + max_nonzeros * idx,
                                    row_offsets.data_ptr<int>() + (m + 1) * batch_index, 
                                    column_indices.data_ptr<int>() + column_indices_tracker,
                                    dense.data_ptr<float>() + k * n * idx,
                                    output.data_ptr<float>() + m * n * idx, 
                                    stream));
        
    }

    return output;
}
