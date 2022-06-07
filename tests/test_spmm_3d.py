import numpy as np
import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time

def mm(sparse, dense, replicaiton, m, k, n):
    return torch.matmul(sparse, dense)

if __name__ == "__main__":
    r, m, k, n, sparsity = 256, 72, 64, 72, 0.90
    
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

    for _ in range(30):
        start = time.time()
        sparse_result = torch_sputnik.spmm(m, k, lhs, topology.row_indices, topology.row_offsets, topology.column_indices, rhs)
        end = time.time()
        sparse_time = end - start

        left = torch.from_numpy(lhs_np).to(torch.float32).cuda()

        start = time.time()
        dense_result = mm(left, rhs, r, m, k, n)
        end = time.time()
        dense_time = end - start
        print(f'Sparse/Dense Time: {sparse_time / dense_time}')

        # if ((sparse_result.view(r, m, n) - dense_result) < 1e4).sum().item() == r * m * n:
        #     print("Results match")
        # else:
        #     print("Results don't match")