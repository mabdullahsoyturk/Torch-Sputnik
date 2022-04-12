#include <torch/extension.h>

// Forward Declaration
void LaunchSpmm(int m, int k, int n, int nonzeros,
                const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix);

void TorchSpmm(torch::Tensor sparse_matrix, 
               torch::Tensor dense_matrix,
               torch::Tensor output_matrix);

void Test();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &LaunchSpmm, "SpMM (CUDA)");
  m.def("torch_spmm", &TorchSpmm, "SpMM (Torch)");
  m.def("test", &Test, "Test Function");
}
