import torch
import torch_sputnik
from utils.util import *

def tensor_spmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    dense = torch.arange(1,nnz + 1, dtype=torch.float32).view(k, n).cuda()

    result = torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, dense)

    print(result)

if __name__ == "__main__":
    tensor_spmm(8, 8, 8, 64)
