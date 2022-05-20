import torch
import numpy as np

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

def diffsort_3d(offsets, m, replication):
     indices_3d = torch.Tensor([]).cuda()
     offset_index_start = 0
     offset_index_end = m + 1

     for idx in range(replication):
          indices = diffsort(offsets[offset_index_start:offset_index_end])
          indices_3d = torch.cat([indices_3d, indices])  

     return indices_3d

def dense_to_sparse_3d(dense):
    replication = dense.size(0)

    values_3d, row_indices_3d, row_offsets_3d, column_indices_3d, nnz_3d = dense_to_sparse(dense[0, :, :])

    for idx in range(1, replication):
        values, row_indices, row_offsets, column_indices, nnz = dense_to_sparse(dense[idx, :, :])

        values_3d = torch.cat([values_3d, values])
        row_indices_3d = torch.cat([row_indices_3d, row_indices])
        row_offsets_3d = torch.cat([row_offsets_3d, row_offsets])
        column_indices_3d = torch.cat([column_indices_3d, column_indices])
        nnz_3d = torch.cat([nnz_3d, nnz])

    return values_3d, row_indices_3d, row_offsets_3d, column_indices_3d, nnz_3d

if __name__ == "__main__":
     #dense = torch.arange(1, 65, dtype=torch.float32).view(8, 8)
     #values, row_indices, row_offsets, column_indices = dense_to_sparse(dense)
 
     #print(dense)
     #print(values)
     #print(row_indices)
     #print(row_offsets)
     #print(column_indices)
     
     dense = torch.Tensor([
     [
          [1,2,3],
          [1,0,0],
          [0,1,2]
     ],
     [
          [1,2,3],
          [1,4,5],
          [0,1,2]
     ],
     [
          [1,2,3],
          [1,0,0],
          [0,1,2]
     ]
     ])
     values, row_indices, row_offsets, column_indices = dense_to_sparse(dense[1,:,:])

     print(dense[1, :, :])
     print(values)
     print(row_indices)
     print(row_offsets)
     print(column_indices)