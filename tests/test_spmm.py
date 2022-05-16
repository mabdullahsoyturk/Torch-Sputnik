import torch
import torch_sputnik
from utils.util import *

def mm(sparse, dense, m, k, n):
    return torch.matmul(sparse, dense)

def spmm(sparse, dense, m, k, n):
    values, row_indices, row_offsets, column_indices, nnzs = dense_to_sparse(sparse)
    result = torch_sputnik.spmm(m, k, n, nnzs, row_indices, values, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    m, k, n = 64, 64, 64
    #sparse = torch.arange(1, (m * k) + 1, dtype=torch.float32).view(m, k).cuda()
    #dense  = torch.arange(1, (k * n) + 1, dtype=torch.float32).view(k, n).cuda()
    sparse = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()
    dense = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()

    sparse_result = spmm(sparse, dense, m, k, n)
    dense_result = mm(sparse, dense, m, k, n)

    #print(sparse_result)
    #print(dense_result)

    print(sparse_result.size())
    print(sparse_result)
    print(dense_result.size())
    print(dense_result)
    print((sparse_result == dense_result).sum().item())
    print(sparse_result - dense_result)
    print(((sparse_result - dense_result) < 1e4).sum().item())