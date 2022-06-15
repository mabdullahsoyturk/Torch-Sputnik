import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time

def mm(sparse, lhs_matrix, rhs_matrix, m, k, n):
    result = torch.matmul(lhs_matrix, rhs_matrix.transpose(-2, -1))
    
    #result.masked_fill_(sparse == torch.tensor(0), 0)

    return result

if __name__ == "__main__":
    m, k, n, sparsity = 72, 64, 72, 0.0

    # Helpers to set up the matrices.
    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([m, k])
    rhs_np = initializer([n, k])
    output_np = connector(np.ones([m, n]))

    topology = sparse_matrix.SparseTopology(mask=output_np)
    lhs = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()
    
    sparse_result = torch_sputnik.sddmm(m, n, topology.row_indices, topology.row_offsets, topology.column_indices, lhs, rhs)
    dense_result = mm(torch.from_numpy(output_np).cuda(), lhs, rhs, m, k, n)

    if ((sparse_result.view(m, n) - dense_result) < 1e-4).sum().item() == m * n:
         print("Results match")
    else:
        print("Results don't match")