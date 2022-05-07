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
