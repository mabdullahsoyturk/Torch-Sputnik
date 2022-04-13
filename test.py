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


def tensor_spmm():
    a = torch.arange(1,65, dtype=torch.float32).view(8,8)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    print("Values:")
    print(values.dtype)
    print("Row Indices:")
    print(row_indices.dtype)
    print("Row Offsets:")
    print(row_offsets.dtype)
    print("Column Indices:")
    print(column_indices.dtype)

    b = torch.arange(1,65).view(8,8).cuda().to(torch.float32)
    print(b.dtype)
    c = torch.zeros((8,8)).cuda()
    print(c.dtype)

    result = torch_sputnik.tensor_spmm(8,8,8,64, row_indices, values, row_offsets, column_indices, b, c)

    print("Result:")
    print(result)

def torch_spmm():
     a = torch.arange(1,65).view(8,8)
     b = torch.arange(1,65).view(8,8).cuda()
     c = torch.zeros((8,8)).cuda()

     result = torch_sputnik.torch_spmm(a,b,c)

if __name__ == "__main__":
    tensor_spmm()
    #torch_spmm()
