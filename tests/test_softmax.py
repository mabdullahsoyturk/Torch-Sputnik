import torch
import torch_sputnik
from utils.util import *

def softmax(sparse, lhs_matrix, rhs_matrix, m, k, n):
    result = torch.matmul(lhs_matrix, rhs_matrix.t())
    result.masked_fill_(sparse == torch.tensor(0), 0)

    softmax = torch.nn.Softmax(dim=-1)
    result = softmax(result)

    return result

def sparse_softmax(sparse, lhs_matrix, rhs_matrix, m, k, n):
    _, row_indices, row_offsets, column_indices = dense_to_sparse(sparse)

    output_values = torch_sputnik.sddmm(m, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)
    result = torch_sputnik.softmax(output_values, row_indices, row_offsets, column_indices)

    return result

if __name__ == "__main__":
    m, k, n = 64, 64, 64

    sparse = torch.rand((m * n), dtype=torch.float32).view(m, n).cuda()
    lhs_matrix = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()
    rhs_matrix = torch.rand((k * n), dtype=torch.float32).view(m, k).cuda()

    sparse_result = sparse_softmax(sparse, lhs_matrix, rhs_matrix, m, k, n)
    dense_result = softmax(sparse, lhs_matrix, rhs_matrix, m, k, n)

    print(sparse_result.size())
    print(sparse_result)
    print(dense_result.size())
    print(dense_result)

    if ((sparse_result.view(m, n) - dense_result) < 1e4).sum().item() == m * n:
        print("Results match")
    else:
        print("Results don't match")
