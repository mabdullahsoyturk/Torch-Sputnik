#include <torch/extension.h>

// Forward Declaration
torch::Tensor spmm(int m, int k,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor dense_matrix);

/*torch::Tensor spmm_graph(int m, int k, int n,
                    torch::Tensor values, 
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets, 
                    torch::Tensor column_indices,
                    torch::Tensor dense_matrix);*/

torch::Tensor sddmm(int m, int n,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor lhs_matrix,
                    torch::Tensor rhs_matrix);

/*torch::Tensor sddmm_graph(int m, int k, int n,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices,
                    torch::Tensor lhs_matrix,
                    torch::Tensor rhs_matrix);*/

torch::Tensor softmax(torch::Tensor values,
                    torch::Tensor row_indices,
                    torch::Tensor row_offsets,
                    torch::Tensor column_indices);

void csr_transpose(
                   int m, int n,
                   torch::Tensor values, 
                   torch::Tensor row_offsets,
                   torch::Tensor column_indices,
                   torch::Tensor output_values,
                   torch::Tensor output_row_offsets,
                   torch::Tensor output_column_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &spmm, "Sparse Matrix Matrix Multiplication: AxB");
  //m.def("spmm_graph", &spmm_graph, "Sparse Matrix Matrix Multiplication: AxB with CUDA Graphs");
  m.def("sddmm", &sddmm, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  //m.def("sddmm_graph", &sddmm_graph, "Sampled Dense Dense Matrix Multiplication: (AxB).C = D");
  m.def("softmax", &softmax, "Computes softmax function across the last dim of a sparse matrix");
  m.def("csr_transpose", &csr_transpose, "Transpose sparse matrix");
}
