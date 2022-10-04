#include <torch/extension.h>

// Forward Declarations
torch::Tensor spmm(int m, int k,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor dense_matrix);

torch::Tensor spmm_many_mask(int b, int m, int k,
                torch::Tensor nonzeros,    
                torch::Tensor values, 
                torch::Tensor row_indices,
                torch::Tensor row_offsets, 
                torch::Tensor column_indices,
                torch::Tensor dense);

torch::Tensor left_spmm(int m, int k,
               torch::Tensor values, 
               torch::Tensor row_indices,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix);

torch::Tensor left_spmm_graph(int m, int k,
               torch::Tensor values, 
               torch::Tensor row_indices,
               torch::Tensor row_offsets, 
               torch::Tensor column_indices,
               torch::Tensor dense_matrix);

torch::Tensor spmm_bias_relu(int m, int k,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor bias,
                    torch::Tensor dense_matrix);

torch::Tensor spmm_graph(int m, int k,
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

torch::Tensor sddmm_graph(int m, int n,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor lhs_matrix,
                    torch::Tensor rhs_matrix);

torch::Tensor sddmm_many_mask(int b, int m, int n,
                              torch::Tensor nonzeros,
                              torch::Tensor row_indices,
                              torch::Tensor row_offsets,
                              torch::Tensor column_indices,
                              torch::Tensor lhs_matrix,
                              torch::Tensor rhs_matrix);

torch::Tensor sparse_softmax(torch::Tensor values,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices);

torch::Tensor sparse_softmax_many_mask(int b, int m,
                                        torch::Tensor nonzeros,
                                        torch::Tensor values,
                                        torch::Tensor row_indices,
                                        torch::Tensor row_offsets,
                                        torch::Tensor column_indices);

std::vector<torch::Tensor> csr_transpose(
                   int m, int n,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices);

std::vector<torch::Tensor> csr_transpose_many_mask(
                   int b, int m, int n,
                    torch::Tensor nonzeros,
                    torch::Tensor values, 
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Matrix Multiplication: AxB");
  m.def("spmm_many_mask", &spmm_many_mask, "Sparse Matrix Matrix Multiplication: AxB");
  m.def("left_spmm", &left_spmm, "Sparse Matrix Matrix Multiplication: A(2d)xB(3d)");
  m.def("left_spmm_graph", &left_spmm_graph, "Sparse Matrix Matrix Multiplication with Cuda Graph: A(2d)xB(3d)");
  m.def("spmm_bias_relu", &spmm_bias_relu, "RELU(Sparse Matrix Matrix Multiplication + bias): RELU(AxB + bias)");
  m.def("spmm_graph", &spmm_graph, "Sparse Matrix Matrix Multiplication: AxB with CUDA Graph");
  m.def("sddmm", &sddmm, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  m.def("sddmm_graph", &sddmm_graph, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D with CUDA Graph");
  m.def("sddmm_many_mask", &sddmm_many_mask, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D with multiple masks");
  m.def("sparse_softmax", &sparse_softmax, "Computes softmax function across the last dim of a sparse matrix");
  m.def("sparse_softmax_many_mask", &sparse_softmax_many_mask, "Computes softmax function across the last dim of a sparse matrix");
  m.def("csr_transpose", &csr_transpose, "Transpose sparse matrix");
  m.def("csr_transpose_many_mask", &csr_transpose_many_mask, "Transpose sparse matrix");
}
