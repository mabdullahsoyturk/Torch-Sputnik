import torch
import torch_sputnik
from utils.util import *

def mm(sparse, dense, m, k, n):
    return torch.matmul(sparse, dense)

def spmm(sparse, dense, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(sparse)
    result = torch_sputnik.spmm(m, k, n, values, row_indices, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    m, k, n = 32, 32, 32
    sparse = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()
    dense = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()

    sparse_result = spmm(sparse, dense, m, k, n)
    dense_result = mm(sparse, dense, m, k, n)

    print(sparse_result)
    print(dense_result)
    
    if ((sparse_result - dense_result) < 1e4).sum().item() == m * n:
        print("Results match")
    else:
        print("Results don't match")