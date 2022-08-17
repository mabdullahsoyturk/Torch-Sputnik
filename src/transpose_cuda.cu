#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cusparse.h>
#include "error_check.h"

torch::Tensor allocate_transpose_workspace(cusparseHandle_t* handle,
        int m, int n, int nonzeros, 
        torch::Tensor values, 
        torch::Tensor row_offsets,
        torch::Tensor column_indices, 
        torch::Tensor output_values, 
        torch::Tensor output_row_offsets,
        torch::Tensor output_column_indices
    ) {

    // Calculate the buffer size.
    size_t buffer_size = 0;
    CUSPARSE_CALL(cusparseCsr2cscEx2_bufferSize(
        *handle, m, n, nonzeros, 
        values.data_ptr<float>(), 
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(), 
        output_values.data_ptr<float>(), 
        output_row_offsets.data_ptr<int>(), 
        output_column_indices.data_ptr<int>(),
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size));

    // Allocate the temporary buffer. Round up to the nearest float for the size of the buffer.
    int buffer_size_signed = (buffer_size + sizeof(float) - 1) / sizeof(float);
    
    auto options = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA, values.device().index());

    torch::Tensor workspace = torch::zeros({buffer_size_signed}, options);

    return workspace;
}

void csr_transpose(int m, int n,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices,
                   torch::Tensor output_values,
                   torch::Tensor output_row_offsets,
                   torch::Tensor output_column_indices) {
    /*--- CHECKS ---*/
    assert(values.dim() == 1 || values.dim() == 2); // Values should have 1 or 2 dimensions
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    assert(values.size(0) == column_indices.size(0)); // Expected same number of values and indices
    assert(row_offsets.size(0) == m + 1); // Expected m+1 row offsets

    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();

    int nonzeros = values.size(-1);

    torch::Tensor workspace = allocate_transpose_workspace(&handle, m, n, nonzeros, 
                                                        values, 
                                                        row_offsets, 
                                                        column_indices, 
                                                        output_values, 
                                                        output_row_offsets, 
                                                        output_column_indices);

    // Launch the kernel.
    CUSPARSE_CALL(cusparseCsr2cscEx2(
        handle, m, n, nonzeros, 
        values.data_ptr<float>(), 
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(), 
        output_values.data_ptr<float>(), 
        output_row_offsets.data_ptr<int>(), 
        output_column_indices.data_ptr<int>(),
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, workspace.data_ptr<float>()));
}