#include <torch/extension.h>

// Forward Declaration
torch::Tensor spmm(int m, int k, int n, int nonzeros,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor dense_matrix,
                    torch::Tensor bias,
                    torch::Tensor output_matrix);

torch::Tensor replicated_spmm(int replication, int m, int k, int n, int nonzeros,
               torch::Tensor row_indices, 
               torch::Tensor values,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix, 
               torch::Tensor bias,
               torch::Tensor output_matrix);

torch::Tensor sddmm(int m, int k, int n, int nonzeros,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor lhs_matrix,
                    torch::Tensor rhs_matrix,
                    torch::Tensor output_values);

torch::Tensor softmax(int m, int n, int nonzeros,
                    torch::Tensor values,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor output_values);

void csr_transpose(
                   int m, int n, int nonzeros,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices,
                   torch::Tensor output_values,
                   torch::Tensor output_row_offsets,
                   torch::Tensor output_column_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Matrix Multiplication: AxB");
  m.def("replicated_spmm", &replicated_spmm, "Replicated Sparse Matrix Matrix Multiplication for 3d tensors");
  m.def("sddmm", &sddmm, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  m.def("softmax", &softmax, "Computes softmax function across the last dim of a sparse matrix");
  m.def("csr_transpose", &csr_transpose, "Transpose sparse matrix");
}
