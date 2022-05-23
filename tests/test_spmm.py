import torch
import torch_sputnik
#from utils.util import *

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().data.clone().to(torch.int32)
     row_indices = diffsort(row_offsets).to(torch.int32)
     column_indices = csr.col_indices().data.clone().to(torch.int32)

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True)

def mm(sparse, dense, m, k, n):
    return torch.matmul(sparse, dense)

def spmm(sparse, dense, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(sparse)
    result = torch_sputnik.spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    m, k, n = 11, 64, 11
    sparse = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()
    dense = torch.rand((k * n), dtype=torch.float32).view(k, n).cuda()

    sparse_result = spmm(sparse, dense, m, k, n)
    dense_result = mm(sparse, dense, m, k, n)

    print(sparse_result)
    print(dense_result)
    
    if ((sparse_result - dense_result) < 1e4).sum().item() == m * n:
        print("Results match")
    else:
        print("Results don't match")