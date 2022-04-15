#include <cusparse.h>

#define CUSPARSE_CALL(code)                                        \
  do {                                                             \
    cusparseStatus_t status = code;                                \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) << "CuSparse Error"; \
  } while (0)

}

torch::Tensor AllocateTransposeWorkspace(
        int m, int n, int nonzeros, 
        torch::Tensor values, 
        torch::Tensor row_offsets,
        torch::Tensor column_indices, 
        torch::Tensor output_values, 
        torch::Tensor output_row_offsets,
        torch::Tensor output_column_indices
    ) {

    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    // Calculate the buffer size.
    size_t buffer_size = 0;
    CUSPARSE_CALL(cusparseCsr2cscEx2_bufferSize(
        stream, m, n, nonzeros, 
        values.data_ptr<float>(), 
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(), 
        output_values.data_ptr<float>(), 
        output_row_offsets.data_ptr<int>(), 
        output_column_indices.data_ptr<int>(),
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size));

    // Allocate the temporary buffer. Round up to the nearest float for the size of the buffer.
    int64 buffer_size_signed = (buffer_size + sizeof(float) - 1) / sizeof(float);
    
    auto options = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA, 0);

    torch::Tensor workspace = torch::zeros(buffer_size_signed, options);

    return workspace;
}

void csr_transpose(int m, int n, int nonzeros,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices,
                   torch::Tensor output_values,
                   torch::Tensor output_row_offsets,
                   torch::Tensor output_column_indices,
                   torch::Tensor workspace) {

    at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = torch_stream.stream();

    // Launch the kernel.
    CUSPARSE_CALL(cusparseCsr2cscEx2(
        stream, m, n, nonzeros, 
        values.data_ptr<float>(), 
        row_offsets.data_ptr<int>(),
        column_indices.data_ptr<int>(), 
        output_values.data_ptr<float>(), 
        output_row_offsets.data_ptr<int>(), 
        output_column_indices.data_ptr<int>(),
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, workspace.data_ptr<float>()));
}