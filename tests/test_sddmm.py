import torch
import torch_sputnik
from utils.util import *

def tensor_sddmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    lhs_matrix = torch.arange(1,nnz + 1).view(k, n).cuda().to(torch.float32)
    print(lhs_matrix)
    rhs_matrix = torch.arange(1,nnz + 1).view(k, n).cuda().to(torch.float32)
    print(rhs_matrix)

    output_values = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, values)

    print(output_values.size())
    print(output_values)

if __name__ == "__main__":
    tensor_sddmm(8, 8, 8, 64)
