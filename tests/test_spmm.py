import torch
import torch_sputnik
from utils.util import *

def mm(m, k, n, nnz):
    sparse = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k).cuda()
    dense = torch.arange(1,nnz + 1, dtype=torch.float32).view(k, n).cuda()

    return torch.matmul(sparse, dense)

def spmm(m, k, n, nnz):
    a = torch.arange(1, m * k + 1, dtype=torch.float32).view(m, k).cuda()
    values, row_indices, row_offsets, column_indices, nnzs = dense_to_sparse(a)

    dense = torch.arange(1, k * n + 1, dtype=torch.float32).view(k, n).cuda()

    result = torch_sputnik.spmm(m, k, n, nnzs, row_indices, values, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    sparse_result = spmm(8, 8, 8, 64)
    dense_result = mm(8, 8, 8, 64)

    print(sparse_result)
    print(dense_result)

    print(sparse_result == dense_result)