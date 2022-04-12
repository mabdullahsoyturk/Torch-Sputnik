#include <torch/extension.h>

// Forward Declaration
void TorchSpmm(torch::Tensor sparse_matrix, 
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix);

torch::Tensor TensorSpmm(int m, int k, int n, int nonzeros,
                torch::Tensor values, torch::Tensor row_indices,
                torch::Tensor row_offsets, torch::Tensor column_indices,
                torch::Tensor dense_matrix,
                torch::Tensor output_matrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_spmm", &TorchSpmm, "SpMM (Torch)");
  m.def("tensor_spmm", &TensorSpmm, "SpMM with PyTorch Tensors");
}
