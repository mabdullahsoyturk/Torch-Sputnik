import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time
import numpy as np

def mm(sparse, dense, m, k, n):
    return torch.matmul(sparse, dense)

if __name__ == "__main__":
    m, k, n, sparsity = 72, 64, 72, 0.9
    
    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    lhs_np = connector(initializer([m, k]))
    rhs_np = initializer([k, n])

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}')

    topology = sparse_matrix.SparseMatrix(matrix=lhs_np)
    lhs = torch.from_numpy(lhs_np[lhs_np != 0]).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    print(f'\nlhs: {lhs.size()}, rhs: {rhs.size()}')

    sparse_result = torch_sputnik.spmm(m, k, lhs, topology.row_indices, topology.row_offsets, topology.column_indices, rhs)
    print(sparse_result)

    left = torch.from_numpy(lhs_np).to(torch.float32).cuda()

    dense_result = mm(left, rhs, m, k, n)
    print(dense_result)

    if (torch.abs(sparse_result - dense_result) < 1e-2).sum() == m * n:
        print("Output matches")
    else:
        print("Doesn't match")