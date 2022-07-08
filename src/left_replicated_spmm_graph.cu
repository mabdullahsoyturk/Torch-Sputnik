#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "error_check.h"

torch::Tensor left_spmm_graph(int m, int k,
               torch::Tensor values, 
               torch::Tensor row_indices,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix) {
    //CHECK_INPUT(values);
    CHECK_INPUT(row_indices);
    CHECK_INPUT(row_offsets);
    CHECK_INPUT(column_indices);
    //CHECK_INPUT(dense_matrix);

    int nonzeros = column_indices.size(-1);
    int dim_offset = dense_matrix.dim() - 2;
    int n = dense_matrix.size(dim_offset + 1);
    int replication = dim_offset == 1 ? dense_matrix.size(0) : 1;

    auto options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());

    torch::Tensor out = replication == 1 ? torch::zeros({m, n}, options) : torch::zeros({replication, m, n}, options);

    // Cuda Graph related part
    cudaStream_t streamForGraph;
    CUDA_CALL(cudaStreamCreate(&streamForGraph));

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    
    CUDA_CALL(cudaStreamBeginCapture(streamForGraph, cudaStreamCaptureModeGlobal));

    for (int idx = 0; idx < replication; ++idx) {
      CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, 
                                  row_indices.data_ptr<int>(), 
                                  values.data_ptr<float>(),
                                  row_offsets.data_ptr<int>(), 
                                  column_indices.data_ptr<int>(),
                                  dense_matrix.data_ptr<float>() + k * n * idx,
                                  out.data_ptr<float>() + m * n * idx, 
                                  streamForGraph));
    }

    CUDA_CALL(cudaStreamEndCapture(streamForGraph, &graph));
    CUDA_CALL(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    CUDA_CALL(cudaGraphLaunch(instance, streamForGraph));
    cudaStreamSynchronize(streamForGraph);

    return out;
}
