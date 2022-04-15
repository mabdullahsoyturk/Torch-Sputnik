#include <torch/extension.h>

// Forward Declaration
torch::Tensor TensorSpmm(int m, int k, int n, int nonzeros,
                torch::Tensor values, 
                torch::Tensor row_indices,
                torch::Tensor row_offsets, 
                torch::Tensor column_indices,
                torch::Tensor dense_matrix,
                torch::Tensor output_matrix);

torch::Tensor TensorSddmm(int m, int k, int n, int nonzeros,
                          torch::Tensor row_indices,
                          torch::Tensor row_offsets,
                          torch::Tensor column_indices,
                          torch::Tensor lhs_matrix,
                          torch::Tensor rhs_matrix,
                          torch::Tensor output_values);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_spmm", &TensorSpmm, "SpMM with PyTorch Tensors");
  m.def("tensor_sddmm", &TensorSddmm, "SDDMM with PyTorch Tensors");
}
