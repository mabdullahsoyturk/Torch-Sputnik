import numpy as np
import torch
import torch.nn as nn
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time

def mm(sparse, dense, replicaiton, m, k, n):
    return torch.matmul(sparse, dense)

if __name__ == "__main__":
    r, m, k, n, sparsity = 256, 72, 64, 72, 0.9
    criterion = nn.MSELoss()
    
    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    mask = connector(initializer([m, k]))
    mask[mask != 0] = 1.0

    print(np.expand_dims(mask, axis=0).shape)

    lhs_np = np.expand_dims(mask, axis=0) * initializer([r, m, k])
    rhs_np = initializer([r, k, n])

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}, mask_np: {mask.shape}')

    topology = sparse_matrix.SparseTopology(mask=mask)
    lhs = torch.from_numpy(np.reshape(lhs_np[lhs_np != 0], [r, -1])).to(torch.float32).cuda().requires_grad_(True)
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    print(f'\nlhs: {lhs.size()}, rhs: {rhs.size()}')

    sparse_result = torch_sputnik.spmm(m, k, lhs, topology.row_indices, topology.row_offsets, topology.column_indices, rhs)

    left = torch.from_numpy(lhs_np).to(torch.float32).cuda()

    dense_result = mm(left, rhs, r, m, k, n)

    print(f'Sparse Result Size: {sparse_result.size()}, Dense Result Size: {dense_result.size()}')
    if ((sparse_result - dense_result) < 1e-2).sum().item() == r * m * n:
        print("Results match")
    else:
        print("Results don't match")