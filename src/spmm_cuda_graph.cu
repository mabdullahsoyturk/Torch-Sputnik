#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor spmm_graph(int m, int k, int n, int nonzeros,
               torch::Tensor row_indices, 
               torch::Tensor values,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);
    int dim_offset = dense_matrix.dim() - 2;
    int replication = dim_offset == 1 ? dense_matrix.size(0) : 1;

    torch::Tensor out = torch::zeros({replication, m, n}, options);

    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for (int idx = 0; idx < replication; ++idx) {
        CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, 
                                    row_indices.data_ptr<int>() + m * idx, 
                                    values.data_ptr<float>() + nonzeros * idx,
                                    row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                    column_indices.data_ptr<int>() + nonzeros * idx,
                                    dense_matrix.data_ptr<float>() + k * n * idx,
                                    out.data_ptr<float>() + m * n * idx, 
                                    stream));
    }

    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaGraphLaunch(instance, stream);
    
    cudaStreamSynchronize(stream);
    
    return out;
}
