import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix

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

def mm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n):
    result = torch.matmul(lhs_matrix, rhs_matrix.transpose(-2, -1))
    
    result.masked_fill_(sparse == torch.tensor(0), 0)

    return result

if __name__ == "__main__":
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0

    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([r, m, k])
    rhs_np = initializer([r, n, k])
    output_np = connector(np.ones([m, n]))

    output_topology = sparse_matrix.SparseTopology(mask=output_np)
    lhs = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()
    
    sparse_result = torch_sputnik.sddmm(m, n, output_topology.row_indices, output_topology.row_offsets, output_topology.column_indices, lhs, rhs)

    dense_result = mm(torch.from_numpy(output_np).cuda(), lhs, rhs, r, m, k, n)

    if ((sparse_result.view(r, m, n) - dense_result) < 1e4).sum().item() == r * m * n:
        print("Results match")
    else:
        print("Results don't match")