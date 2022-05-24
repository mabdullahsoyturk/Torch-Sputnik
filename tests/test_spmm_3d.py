import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix

def mm(sparse, dense, replicaiton, m, k, n):
    return torch.matmul(sparse, dense)

if __name__ == "__main__":
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0
    
    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    mask = connector(initializer([m, k]))
    mask[mask != 0] = 1.0

    lhs_np = np.expand_dims(mask, axis=0) * initializer([r, m, k])
    rhs_np = initializer([r, k, n])

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}, mask_np: {mask.shape}')

    topology = sparse_matrix.SparseTopology(mask=mask)
    lhs = torch.from_numpy(np.reshape(lhs_np[lhs_np != 0], [r, -1])).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    print(f'\nlhs: {lhs.size()}, rhs: {rhs.size()}')

    sparse_result = torch_sputnik.spmm(m, k, lhs, topology.row_indices, topology.row_offsets, topology.column_indices, rhs)

    dense_result = mm(lhs.reshape(r, m, k), rhs, r, m, k, n)

    if ((sparse_result.view(r, m, n) - dense_result) < 1e4).sum().item() == r * m * n:
        print("Results match")
    else:
        print("Results don't match")