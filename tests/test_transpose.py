import torch
import torch_sputnik
from utils.util import *

def sparse_transpose(sparse, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(sparse)
    output_values = values.clone()
    output_row_offsets = row_offsets.clone()
    output_column_indices = column_indices.clone()

    torch_sputnik.csr_transpose(m, n, values, row_offsets, column_indices, output_values, output_row_offsets, output_column_indices)

    return output_values

if __name__ == "__main__":
    m, k, n = 64, 64, 64
    sparse = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()

    sparse_result = sparse_transpose(sparse, m, k, n).reshape(k, m)

    print(sparse_result)
    print(sparse.t())

    if ((sparse_result - sparse.t()) < 1e4).sum().item() == m * n:
        print("Results match")
    else:
        print("Results don't match")