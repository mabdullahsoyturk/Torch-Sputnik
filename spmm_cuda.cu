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
void LaunchSpmm(int m, int k, int n, int nonzeros,
                const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix) {
  // NOTE: Passing nullptr as bias will execute the standard spmm w/ no bias or relu.
  auto stream = c10::cuda::getDefaultCUDAStream(0);
  CUDA_CALL(sputnik::CudaSpmmBiasRelu(m, k, n, nonzeros, row_indices, values,
                                      row_offsets, column_indices, dense_matrix,
                                      bias, output_matrix, stream));
}

// CUDA kernel launcher.
void TorchSpmm(torch::Tensor sparse_matrix, 
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix) {
  // NOTE: Passing nullptr as bias will execute the standard spmm w/ no bias or relu.
  const float* bias = NULL;
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  printf("Set device and create stream\n");
  int m = 8;
  int n = 8;
  int k = 8;
  int nnz = 64;
  
  // Values of the sparse matrix
  float values[nnz];
  for(int i = 0; i < nnz; i++) {
    values[i] = (float)(i + 1);
  }
  
  // Row indices
  int row_indices[m];
  int index = 0;
  for(int i = 0; i < m; i++) {
    row_indices[i] = index;
    index += m;
  }

  int row_offsets[m + 1];
  index = 0;
  for(int i = 0; i < m + 1; i++) {
    row_offsets[i] = index;
    index += m;
  }

  int column_indices[nnz];
  int column_index = 0;
  for(int i = 0; i < nnz; i++) {
    column_indices[i] = column_index;
    column_index++;

    column_index = column_index % n;
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
  for(int i = 0; i < m * n; i++) {
    printf("Index %d: %f\n", i, output[i]);
  }
  cudaStreamDestroy(stream);
}

__global__ void test_kernel(float* a, float* b, float* c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x]; 
}

void Test() {
  const float a[] = { 1.0f,  2.0f,  3.0f,  4.0f,
                          5.0f,  6.0f,  7.0f,  8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f};

  float* d_a;
  CUDA_CALL(cudaMalloc(&d_a, 12 * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_a, a, 12 * sizeof(float), cudaMemcpyHostToDevice));

  const float b[] = { 1.0f,  2.0f,  3.0f,  4.0f,
                          5.0f,  6.0f,  7.0f,  8.0f,
                          9.0f, 10.0f, 11.0f, 12.0f};

  float* d_b;
  CUDA_CALL(cudaMalloc(&d_b, 12 * sizeof(float)));
  CUDA_CALL(cudaMemcpy(d_b, b, 12 * sizeof(float), cudaMemcpyHostToDevice));

  float* d_c;
  CUDA_CALL(cudaMalloc(&d_c, 12 * sizeof(float)));

  printf("Before the kernel\n");
  test_kernel<<<1, 12>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();
  printf("After the kernel\n");
}