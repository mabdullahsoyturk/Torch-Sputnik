import torch
import torch_sputnik
from utils.util import *

if __name__ == "__main__":
    m, k, n = 3, 3, 3

    sparse = torch.Tensor([
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,4,5],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ]
    ]).cuda()

    dense = torch.Tensor([
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,4,5],
            [0,1,2]
        ],
        [
            [1,2,3],
            [1,0,0],
            [0,1,2]
        ]
    ]).cuda()

    values_3d, row_indices_3d, row_offsets_3d, column_indices_3d, nnz_3d = dense_to_sparse_3d(sparse)

    print(values_3d)
    print(row_indices_3d)
    print(row_offsets_3d)
    print(column_indices_3d)
    print(nnz_3d)

    sparse_result = torch_sputnik.spmm(m, k, n, nnz_3d,
                            row_indices_3d, 
                            values_3d, 
                            row_offsets_3d, 
                            column_indices_3d, 
                            dense
                   )
    
    print(sparse_result)
    dense_result = torch.matmul(sparse, dense)
    print(dense_result)

    if (sparse_result == dense_result).sum() == m * k * n:
        print("Resultl is correct")