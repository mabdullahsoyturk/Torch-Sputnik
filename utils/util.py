import torch
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().detach().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().detach().to(torch.int32)

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

if __name__ == "__main__":
     #dense = torch.arange(1, 65, dtype=torch.float32).view(8, 8)
     #values, row_indices, row_offsets, column_indices = dense_to_sparse(dense)
#
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