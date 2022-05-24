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

def mm(sparse, dense, replicaiton, m, k, n):
    return torch.matmul(sparse, dense)

def spmm(sparse, dense, r, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse_3d(sparse)
    result = torch_sputnik.spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

    return result

if __name__ == "__main__":
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0
    
    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    mask = connector(initializer([m, k]))
    mask[mask != 0] = 1.0

    lhs_np = np.expand_dims(mask, axis=0) * initializer([r, m, k])
    rhs_np = initializer([r, k, n])

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}, mask_np: {mask.shape}')

    topology = sparse_matrix.SparseTopology("topology", mask=mask)
    lhs = torch.from_numpy(np.reshape(lhs_np[lhs_np != 0], [r, -1])).to(torch.float32).cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    sparse_result = torch_sputnik.spmm(m, k, lhs, topology.row_indices, topology.row_offsets, topology.column_indices, rhs)

    dense_result = mm(lhs.reshape(r, m, k), rhs, r, m, k, n)

    if ((sparse_result.view(r, m, n) - dense_result) < 1e4).sum().item() == r * m * n:
        print("Results match")
    else:
        print("Results don't match")