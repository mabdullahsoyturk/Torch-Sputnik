#include <sputnik/sputnik.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cusparse.h>
#include <vector>
#include "error_check.h"

torch::Tensor allocate_transpose_workspace_many(cusparseHandle_t* handle,
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

std::vector<torch::Tensor> csr_transpose_many_mask(int b, int m, int n,
                    torch::Tensor nonzeros,
                    torch::Tensor values, 
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices) {
    /*--- CHECKS ---*/
    assert(row_offsets.dim() == 1); // Row offsets should have 1 dimension
    assert(column_indices.dim() == 1); // Column indices should have 1 dimension
    //assert(row_offsets.size(0) == m + 1); // Expected m+1 row offsets

    cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();

    int replication = values.size(0);

    int num_heads = replication / b;
    int column_indices_out_size = 0;
    int max_nonzeros = -1;

    for(int i = 0; i < nonzeros.size(0); i++) {
        int nonzero = nonzeros[i].item<int>();
        if(nonzero > max_nonzeros) {
            max_nonzeros = nonzero;
        }

        column_indices_out_size += nonzero;
    }

    auto values_options = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());

    auto index_options = torch::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .layout(torch::kStrided)
                                        .device(torch::kCUDA, values.device().index());
    

    torch::Tensor output_values = torch::zeros({replication, max_nonzeros}, values_options);
    torch::Tensor output_row_offsets = torch::zeros({(n + 1) * nonzeros.size(0)}, index_options);
    torch::Tensor output_column_indices = torch::zeros({column_indices_out_size}, index_options);

    std::vector<torch::Tensor> out_vector;
    out_vector.push_back(output_values);
    out_vector.push_back(output_row_offsets);
    out_vector.push_back(output_column_indices);

    for(int idx = 0; idx < replication; idx++) {
        int batch_index = idx / num_heads;
        int nonzero = nonzeros[batch_index].item<int>();
        // (Possibly) get a temporary buffer to work in.
        torch::Tensor workspace = allocate_transpose_workspace_many(&handle, m, n, nonzero, 
                                                            values, 
                                                            row_offsets, 
                                                            column_indices, 
                                                            output_values, 
                                                            output_row_offsets, 
                                                            output_column_indices);

        // Launch the kernel.
        CUSPARSE_CALL(cusparseCsr2cscEx2(
            handle, m, n, nonzero, 
            values.data_ptr<float>() + batch_index * max_nonzeros, 
            row_offsets.data_ptr<int>() + batch_index * (n + 1),
            column_indices.data_ptr<int>(), 
            output_values.data_ptr<float>(), 
            output_row_offsets.data_ptr<int>(), 
            output_column_indices.data_ptr<int>(),
            CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, workspace.data_ptr<float>()));
    }

    return out_vector;
}