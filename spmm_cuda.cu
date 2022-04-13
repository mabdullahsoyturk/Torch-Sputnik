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

// CUDA kernel launcher.
/*void LaunchSpmm(int m, int k, int n, int nonzeros,
                const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix,
                float *output_matrix) {
  auto stream = c10::cuda::getDefaultCUDAStream(0);
  CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, row_indices, values,
                                      row_offsets, column_indices, dense_matrix,
                                      output_matrix, stream));
  cudaDeviceSynchronize();
}*/

torch::Tensor TensorSpmm(int m, int k, int n, int nonzeros,
               torch::Tensor row_indices, torch::Tensor values,
               torch::Tensor row_offsets, torch::Tensor column_indices,
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* _values = values.data_ptr<float>();
    int* _row_indices = row_indices.data_ptr<int>();
    int* _row_offsets = row_offsets.data_ptr<int>();
    int* _column_indices = column_indices.data_ptr<int>();
    float* _dense_matrix = dense_matrix.data_ptr<float>();
    float* _output_matrix = output_matrix.data_ptr<float>();

    CUDA_CALL(sputnik::CudaSpmm(m, k, n, nonzeros, 
                                _row_indices, _values,
                                _row_offsets, _column_indices,
                                _dense_matrix, _output_matrix, stream));
    cudaDeviceSynchronize();
    cudaStreamDestroy(stream);
    
    return output_matrix;
}

// CUDA kernel launcher.
void TorchSpmm(torch::Tensor sparse_matrix, 
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix) {
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  printf("Set device and create stream\n");
  int m = 8;
  int k = 8;
  int n = 8;
  int nnz = 64;
  
  // Values of the sparse matrix
  float values[nnz];
  for(int i = 0; i < nnz; i++) {
    values[i] = (float)(i + 1);
    printf("values[%d]: %f\n", i, values[i]);
  }
  
  int row_offsets[m + 1];
  int index = 0;
  for(int i = 0; i < m + 1; i++) {
    row_offsets[i] = index;
    printf("row_offsets[%d]: %d\n", i, row_offsets[i]);
    index += k;
  }

  int column_indices[nnz];
  for(int i = 0; i < nnz; i++) {
    column_indices[i] = i % k;
    printf("column_indices[%d]: %d\n", i, column_indices[i]);
  }
  // Row indices
  int row_indices[m];
  for(int i = 0; i < m; i++) {
    row_indices[i] = i;
    printf("row_indices[%d]: %d\n", i, row_indices[i]);
  }
  
  float dense[k * n];
  for(int i = 0; i < k * n; i++) {
    dense[i] = (float)(i + 1);
  }

  int* d_row_indices;
  CUDA_CALL(cudaMalloc(&d_row_indices, m * sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_row_indices, row_indices, m * sizeof(int), cudaMemcpyHostToDevice));

  float* d_values;
  CUDA_CALL(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice));

  int* d_row_offsets;
  CUDA_CALL(cudaMalloc(&d_row_offsets, (m + 1) * sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_row_offsets, row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));

  int* d_column_indices;
  CUDA_CALL(cudaMalloc(&d_column_indices, nnz * sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_column_indices, column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
  
  float* d_dense;
  CUDA_CALL(cudaMalloc(&d_dense, k * n * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_dense, dense, k * n * sizeof(float), cudaMemcpyHostToDevice));
  

  float output[m * n];
  for(int i = 0; i < m * n; i++) {
    output[i] = 0;
  }

  float* d_output;
  CUDA_CALL(cudaMalloc(&d_output, m * n * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_output, output, m * n * sizeof(float), cudaMemcpyHostToDevice));

  printf("Moved all data to GPU\n");
 
  CUDA_CALL(sputnik::CudaSpmm(m, k, n, nnz, 
                                      d_row_indices, d_values,
                                      d_row_offsets, d_column_indices, d_dense,
                                      d_output, stream));
  cudaDeviceSynchronize();
  CUDA_CALL(cudaMemcpy(output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost));
  //for(int i = 0; i < m * n; i++) {
  //  printf("Index %d: %f\n", i, output[i]);
  //}
  cudaStreamDestroy(stream);
}
