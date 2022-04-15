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

def tensor_sddmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    lhs_matrix = torch.arange(1,nnz + 1).view(k, n).cuda().to(torch.float32)
    print(lhs_matrix)
    rhs_matrix = torch.arange(1,nnz + 1).view(k, n).cuda().to(torch.float32)
    print(rhs_matrix)

    output_values = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, values)

    print(output_values)

    softmax_output = torch_sputnik.softmax(m, n, nnz, output_values, row_indices, row_offsets, column_indices, output_values)
    print(softmax_output)

if __name__ == "__main__":
    tensor_sddmm(8, 8, 8, 64)
