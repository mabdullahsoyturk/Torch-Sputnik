import numpy as np
import torch
import torch_sputnik

import connectors
import initializers
import sparse_matrix
import time
import math

def dense_attention(batch_size=32, num_heads=8, m=72, k=64):
    query = torch.rand((batch_size, num_heads, m, k))
    key = torch.rand((batch_size, num_heads, m, k))
    value = torch.rand((batch_size, num_heads, m, k))

    start = time.time()
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(64)

    attention_weights = scores.softmax(dim=-1)
        
    representations = torch.matmul(attention_weights, value)
    end = time.time()
    dense_time = end - start
    
    return dense_time

if __name__ == "__main__":
    r, m, k, n, sparsity = 256, 72, 64, 72, 0.9

    connector = connectors.Uniform(sparsity, round_to=4)
    initializer = initializers.Uniform()

    lhs_np = initializer([r, m, k])
    rhs_np = initializer([r, n, k])
    v_np   = initializer([r, m, k])
    output_np = connector(np.ones([m, n]))

    print(f'\nlhs_np: {lhs_np.shape}, rhs_np: {rhs_np.shape}, mask_np: {output_np.shape}')

    topology = sparse_matrix.SparseTopology(mask=output_np)
    q3d = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    k3d = torch.from_numpy(rhs_np).to(torch.float32).cuda()
    v3d = torch.from_numpy(v_np).to(torch.float32).cuda()

    print(f'\nq3d: {q3d.size()}, k3d: {k3d.size()}, v3d: {v3d.size()}, values: {topology.column_indices.size()}')

    ratio = 0.0

    for _ in range(200):
        start = time.time()
        # SDDMM
        scores = torch_sputnik.sddmm(m, n, topology.row_indices, topology.row_offsets, topology.column_indices, q3d, k3d)

        # Softmax
        attention_weights = torch_sputnik.sparse_softmax(scores, topology.row_indices, topology.row_offsets, topology.column_indices)
        
        # SpMM
        intermediate_token_representations = torch_sputnik.spmm(m, n, attention_weights,
                    topology.row_indices, 
                    topology.row_offsets, 
                    topology.column_indices, 
                    v3d
        )
        #print(f'\nq3d: {q3d.size()}, k3d: {k3d.size()}, v3d: {v3d.size()}, row_indices: {topology.row_indices.size()}, row_offsets: {topology.row_offsets.size()}, column_indices: {topology.column_indices.size()}, m: {m}, k: {k}, n: {n}, scores: {scores.size()}, attention_weights: {attention_weights.size()}, intermediate_token_representations: {intermediate_token_representations.size()}')
        end = time.time()

        sparse_time = end - start

        dense_time = dense_attention()

        ratio += sparse_time / dense_time

        print(f'Sparse/Dense Time: {sparse_time / dense_time}')

    print(f'Mean: {ratio / 200}')