#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)

torch::Tensor softmax(int m, n, int nonzeros,
                      torch::Tensor values,
                      torch::Tensor row_indices,
                      torch::Tensor row_offsets,
                      torch::Tensor column_indices,
                      torch::Tensor output_values) {
    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();
    float* _values = values.data_ptr<float>();
    int* _row_indices = row_indices.data_ptr<int>();
    int* _row_offsets = row_offsets.data_ptr<int>();
    int* _column_indices = column_indices.data_ptr<int>();
    float* _output_values = output_values.data_ptr<float>();

    CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzeros, 
                                _values,
                                _row_indices, 
                                _row_offsets, 
                                _column_indices,
                                _output_values, 
                                stream));
    cudaDeviceSynchronize();
    
    return output_matrix;
}
