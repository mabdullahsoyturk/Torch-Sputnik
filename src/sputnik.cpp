#include <torch/extension.h>

// Forward Declaration
torch::Tensor spmm(int m, int k, int n, int nonzeros,
                torch::Tensor values, 
                torch::Tensor row_indices,
                torch::Tensor row_offsets, 
                torch::Tensor column_indices,
                torch::Tensor dense_matrix,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Matrix Multiplication: AxB");
  m.def("sddmm", &sddmm, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  m.def("softmax", &softmax, "Computes softmax function across the last dim of a sparse matrix");
}