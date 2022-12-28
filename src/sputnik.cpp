#include <torch/extension.h>

// Forward Declarations
torch::Tensor spmm(int m, int k,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor dense_matrix);

torch::Tensor left_spmm(int m, int k,
               torch::Tensor values, 
               torch::Tensor row_indices,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix);

torch::Tensor sddmm(int m, int n,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor lhs_matrix,
                    torch::Tensor rhs_matrix);

torch::Tensor sparse_softmax(torch::Tensor values,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices);

std::vector<torch::Tensor> csr_transpose(
                   int m, int n,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Matrix Multiplication: AxB");
  m.def("left_spmm", &left_spmm, "Sparse Matrix Matrix Multiplication: A(2d)xB(3d)");
  m.def("sddmm", &sddmm, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  m.def("sparse_softmax", &sparse_softmax, "Computes softmax function across the last dim of a sparse matrix");
  m.def("csr_transpose", &csr_transpose, "Transpose sparse matrix");
}
