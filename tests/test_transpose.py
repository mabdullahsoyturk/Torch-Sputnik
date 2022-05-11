import torch
import torch_sputnik
from utils.util import *

def transpose(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    b = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    out_values, out_row_indices, out_row_offsets, out_column_indices = dense_to_sparse(b)

    print('Before transpose:')
    print(values.size())
    print(values)

    nonzeros = torch.IntTensor([nnz])
    
    torch_sputnik.csr_transpose(m, n, nonzeros, values, row_offsets, column_indices, out_values, out_row_offsets, out_column_indices)
    
    print('\nAfter transpose:')
    print(out_values.size())
    print(out_values)

if __name__ == "__main__":
    transpose(8, 8, 8, 64)
