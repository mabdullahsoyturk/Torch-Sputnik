import torch
import torch_sputnik
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().clone()
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy())))
     column_indices = csr.col_indices().clone()

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

a = torch.arange(1,65).view(8,8)
values, row_indices, row_offsets, column_indices = dense_to_sparse(a)
print(values)
print(row_indices)
print(row_offsets)
print(column_indices)

b = torch.arange(1,65).view(8,8).cuda()
c = torch.zeros((8,8)).cuda()

result = torch_sputnik.tensor_spmm(8,8,8,64,
                        row_indices, values, row_offsets, column_indices, b, c)

print(result)
