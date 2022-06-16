#pragma once

#include <cusparse.h>

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)

#define CUSPARSE_CALL(code)                                        \
  do {                                                             \
    cusparseStatus_t status = code;                                \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) << "CuSparse Error"; \
  } while (0)

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
