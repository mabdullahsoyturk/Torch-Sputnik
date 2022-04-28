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

def tensor_sddmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)
    b = torch.arange(nnz + 1, (2 * nnz) + 1, dtype=torch.float32).view(m, k)
    values2, row_indices2, row_offsets2, column_indices2 = dense_to_sparse(b)

    c_values = torch.cat((values, values2))
    c_row_indices = torch.cat((row_indices, row_indices2))
    c_row_offsets = torch.cat((row_offsets, row_offsets2))
    c_column_indices = torch.cat((column_indices, column_indices2))
    print(values)
    print(values2)
    print(c_values.view(2, 8, 8).size())

    lhs_matrix = torch.arange(1, (2 * nnz) + 1).view(2, k, n).cuda().to(torch.float32)
    print(lhs_matrix)
    rhs_matrix = torch.arange(1, (2 * nnz) + 1).view(2, k, n).cuda().to(torch.float32)
    print(rhs_matrix)

    output_values = torch_sputnik.replicated_sddmm(2, m, k, n, nnz, c_row_indices, c_row_offsets, c_column_indices, lhs_matrix, rhs_matrix, c_values).view(2, m, n)

    print(output_values.size())
    print(output_values)

if __name__ == "__main__":
    tensor_sddmm(8, 8, 8, 64)
