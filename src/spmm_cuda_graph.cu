#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor spmm_graph(int m, int k, int n,
               torch::Tensor row_indices, 
               torch::Tensor values,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    int nonzeros = column_indices.size(0);
    int dim_offset = dense_matrix.dim() - 2;
    int replication = dim_offset == 1 ? dense_matrix.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, 0)
                                        .requires_grad(true);

    torch::Tensor out = replication == 1 ? torch::zeros({m, n}, options) : torch::zeros({replication, m, n}, options);

    cudaStream_t streamForGraph;
    CUDA_CALL(cudaStreamCreate(&streamForGraph));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    printf("Stream capture is beginning\n");
    CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    printf("Stream capture began\n");

    for (int idx = 0; idx < replication; ++idx) {
        sputnik::CudaSpmm(m, k, n, nonzeros, 
                                    row_indices.data_ptr<int>() + m * idx, 
                                    values.data_ptr<float>() + nonzeros * idx,
                                    row_offsets.data_ptr<int>() + (m + 1) * idx, 
                                    column_indices.data_ptr<int>() + nonzeros * idx,
                                    dense_matrix.data_ptr<float>() + k * n * idx,
                                    out.data_ptr<float>() + m * n * idx, 
                                    stream);
    }
    
    printf("Called all kernels\n");
    CUDA_CALL(cudaStreamEndCapture(stream, &graph));
    printf("Stream capture ended\n");
    
    CUDA_CALL(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("Graph is instantiated\n");
    CUDA_CALL(cudaGraphLaunch(graphExec, streamForGraph));
    printf("Graph launched\n");
    
    CUDA_CALL(cudaStreamSynchronize(streamForGraph));

    CUDA_CALL(cudaGraphExecDestroy(graphExec));
    CUDA_CALL(cudaGraphDestroy(graph));
    CUDA_CALL(cudaStreamDestroy(streamForGraph));
    
    return out;
}
