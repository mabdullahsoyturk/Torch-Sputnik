import torch
import torch_sputnik
import connectors
import initializers
import sparse_matrix
import time
import numpy as np

def softmax(matrix_np):
    # Zero terms should not contribute to the softmax.
    matrix_np[matrix_np == 0] = -1e9

    matrix = torch.from_numpy(matrix_np).to(torch.float32).cuda()

    softmax = torch.nn.Softmax(dim=-1)

    start = time.time()
    result = softmax(matrix)
    end = time.time()
    dense_time = end - start

    return result, dense_time

if __name__ == "__main__":
    m, k, n, sparsity = 72, 64, 72, 0.9

    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    matrix_np = connector(initializer([m, n]))

    print(f'\nmatrix_np: {matrix_np.shape}')

    matrix = sparse_matrix.SparseMatrix(matrix=matrix_np)
    #matrix.values = [matrix.values == 0] = -1e9

    print(f'\nvalues: {matrix.values.size()}, row_indices: {matrix.row_indices.size()}, row_offsets: {matrix.row_offsets.size()}, column_indices: {matrix.column_indices.size()}')

    for _ in range(30):
        start = time.time()
        sparse_result = torch_sputnik.sparse_softmax(matrix.values, matrix.row_indices, matrix.row_offsets, matrix.column_indices)
        end = time.time()
        sparse_time = end - start

        dense_result, dense_time = softmax(matrix_np)

        print(f'Sparse/Dense Time: {sparse_time / dense_time}')
    