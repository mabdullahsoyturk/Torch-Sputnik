import torch
import torch_sputnik
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().detach().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().detach().to(torch.int32)

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

def tensor_spmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    dense = torch.arange(1,nnz + 1).view(k, n).cuda().to(torch.float32)

    result = torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, dense)

    print(result)

if __name__ == "__main__":
    tensor_spmm(8, 8, 8, 64)
