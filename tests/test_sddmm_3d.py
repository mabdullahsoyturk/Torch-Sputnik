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

def mm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n):
    result = torch.matmul(lhs_matrix, rhs_matrix)
    
    result.masked_fill_(sparse == torch.tensor(0), 0)

    return result

def sddmm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n):
    print(sparse.size())
    _, row_indices, row_offsets, column_indices = dense_to_sparse_3d(sparse)
    print(row_indices.size())
    print(row_offsets.size())
    print(column_indices.size())
    print(lhs_matrix.size())
    print(rhs_matrix.size())

    output_values = torch_sputnik.sddmm(m, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)
    print(output_values.size())

    return output_values

if __name__ == "__main__":
    replication, m, k, n = 488, 15, 64, 15

    sparse = torch.rand((replication * m * n), dtype=torch.float32).view(replication, m, n).cuda()
    lhs_matrix = torch.rand((replication * m * k), dtype=torch.float32).view(replication, m, k).cuda()
    rhs_matrix = torch.rand((replication * k * n), dtype=torch.float32).view(replication, k, n).cuda()

    sparse_result = sddmm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n)
    dense_result = mm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n)

    #print(sparse_result.size())
    #print(dense_result.size())

    if ((sparse_result.view(replication, m, n) - dense_result) < 1e4).sum().item() == replication * m * n:
        print("Results match")
    else:
        print("Results don't match")
