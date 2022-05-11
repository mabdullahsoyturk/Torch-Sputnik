import torch
import torch_sputnik
from utils.util import *

def mm(m, k, n, nnz):
    lhs_matrix = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k).cuda()
    rhs_matrix = torch.arange(1, nnz + 1, dtype=torch.float32).view(k, n).cuda()

    return torch.matmul(lhs_matrix, rhs_matrix.t())

def sddmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    mask, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    lhs_matrix = torch.arange(1,nnz + 1, dtype=torch.float32).view(k, n).cuda()
    rhs_matrix = torch.arange(1,nnz + 1, dtype=torch.float32).view(k, n).cuda()

    nonzeros = torch.IntTensor([nnz])

    output_values = torch_sputnik.sddmm(m, k, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, mask)

    return output_values

if __name__ == "__main__":
    sparse_result = sddmm(8, 8, 8, 64)
    dense_result = mm(8, 8, 8, 64)

    print(sparse_result)
    print(dense_result)
    print(sparse_result.view(8,8) == dense_result)
