#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor sddmm_graph(int m, int k, int n,
                           torch::Tensor row_indices,
                           torch::Tensor row_offsets,
                           torch::Tensor column_indices,
                           torch::Tensor lhs_matrix,
                           torch::Tensor rhs_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int nonzeros = column_indices.size(0);
    int dim_offset = lhs_matrix.dim() - 2;
    int replication = dim_offset == 1 ? lhs_matrix.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, lhs_matrix.device().index())
                                        .requires_grad(true);

    torch::Tensor output_values = replication == 1 ? torch::zeros({nonzeros}, options) : torch::zeros({replication, nonzeros}, options);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSddmm(m, k, n, nonzeros, 
                                row_indices.data_ptr<int>() + m * idx, 
                                row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                column_indices.data_ptr<int>() + nonzeros * idx,
                                lhs_matrix.data_ptr<float>() + m * k * idx, 
                                rhs_matrix.data_ptr<float>() + k * n * idx, 
                                output_values.data_ptr<float>() + nonzeros * idx, 
                                stream));
    }
    
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaGraphLaunch(instance, stream);
    
    cudaStreamSynchronize(stream);
    
    return output_values;
}
