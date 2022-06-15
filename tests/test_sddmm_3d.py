import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time

def mm(sparse, lhs_matrix, rhs_matrix, replication, m, k, n):
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

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}, mask_np: {output_np.shape}')

    topology = sparse_matrix.SparseTopology(mask=output_np)
    lhs = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    print(f'\nlhs: {lhs.size()}, rhs: {rhs.size()}')
    
    sparse_result = torch_sputnik.sddmm(m, n, topology.row_indices, topology.row_offsets, topology.column_indices, lhs, rhs)

    dense_result = mm(torch.from_numpy(output_np).cuda(), lhs, rhs, r, m, k, n)

    print(sparse_result.size())
    print(dense_result.size())