import torch
import torch_sputnik
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().clone().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().clone().to(torch.int32)

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

def transpose(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    b = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    out_values, out_row_indices, out_row_offsets, out_column_indices = dense_to_sparse(b)

    print('Before transpose:')
    #print(column_indices.size())
    #print(column_indices)

    #print(row_offsets.size())
    #print(row_offsets)\
    
    print(values.size())
    print(values)
    
    torch_sputnik.csr_transpose(m, n, nnz, values, row_offsets, column_indices, out_values, out_row_offsets, out_column_indices)
    
    print('\nAfter transpose:')
    #print(transpose_out_column_indices.size())
    #print(transpose_out_column_indices)

    #print(transpose_out_row_offsets.size())
    #print(transpose_out_row_offsets)

    print(out_values.size())
    print(out_values)

if __name__ == "__main__":
    transpose(8, 8, 8, 64)
