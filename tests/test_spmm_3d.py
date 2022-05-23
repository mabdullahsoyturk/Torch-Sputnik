import torch
import torch_sputnik
#from utils.util import *

def dense_to_sparse_3d(dense):
    replication = dense.size(0)

    values_3d, row_indices_3d, row_offsets_3d, column_indices_3d = [], [], [], []

    for idx in range(replication):
        values, row_indices, row_offsets, column_indices = dense_to_sparse(dense[idx, :, :])
        values_3d.append(values)
        row_indices_3d.append(row_indices)
        row_offsets_3d.append(row_offsets)
        column_indices_3d.append(column_indices)

    return torch.stack(values_3d), torch.stack(row_indices_3d), torch.stack(row_offsets_3d), torch.stack(column_indices_3d)

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

def mm(sparse, dense, replicaiton, m, k, n):
    return torch.matmul(sparse, dense)

def spmm(sparse, dense, replication, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse_3d(sparse)
    result = torch_sputnik.spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    replication, m, k, n = 584, 11, 11, 64
    sparse = torch.rand((replication * m * k), dtype=torch.float32).view(replication, m, k).cuda()
    dense = torch.rand((replication * k * n), dtype=torch.float32).view(replication, k, n).cuda()

    sparse_result = spmm(sparse, dense, replication, m, k, n)
    dense_result = mm(sparse, dense, replication, m, k, n)

    #print(sparse_result)
    #print(dense_result)
    
    print(((sparse_result - dense_result) < 1e4).sum().item())

    if ((sparse_result - dense_result) < 1e4).sum().item() == replication * m * n:
        print("Results match")
    else:
        print("Results don't match")