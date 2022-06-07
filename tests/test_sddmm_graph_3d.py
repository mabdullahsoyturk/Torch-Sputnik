import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time

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

def mm(sparse, lhs_matrix, rhs_matrix):
    result = torch.matmul(lhs_matrix, rhs_matrix.transpose(-2, -1))
    
    result.masked_fill_(sparse == torch.tensor(0), 0)

    return result

if __name__ == "__main__":
    r, m, k, n, sparsity = 256, 72, 64, 72, 0.9

    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([r, m, k])
    rhs_np = initializer([r, n, k])
    output_np = connector(np.ones([m, n]))

    topology = sparse_matrix.SparseTopology(mask=output_np)
    lhs = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()
    
    start = time.time()
    sparse_result = torch_sputnik.sddmm(m, n, topology.row_indices, topology.row_offsets, topology.column_indices, lhs, rhs)
    end = time.time()
    print(f'Sparse time: {end - start}')

    sparse_time = end - start

    start = time.time()
    dense_result = mm(torch.from_numpy(output_np).cuda(), lhs, rhs)
    end = time.time()

    dense_time = end - start
    print(f'Dense time: {end - start}')

    print(f'Sparse/Dense Time: {sparse_time / dense_time}')

    # if ((sparse_result.view(r, m, n) - dense_result) < 1e4).sum().item() == r * m * n:
    #     print("Results match")
    # else:
    #     print("Results don't match")