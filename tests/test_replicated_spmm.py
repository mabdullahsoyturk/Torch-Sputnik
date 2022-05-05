import torch
import torch_sputnik
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().detach().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().detach().to(torch.int32)

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

def tensor_spmm(m, k, n, nnz):
    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)
    b = torch.arange(nnz + 1, (2 * nnz) + 1, dtype=torch.float32).view(m, k)
    values2, row_indices2, row_offsets2, column_indices2 = dense_to_sparse(b)
    sparse_values = torch.cat((values, values2))
    sparse_row_indices = torch.cat((row_indices, row_indices2))
    sparse_row_offsets = torch.cat((row_offsets, row_offsets2))
    sparse_column_indices = torch.cat((column_indices, column_indices2))

    """
        Sparse Matrix: [
            [
                [1,2,3,4,5,6,7,8],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [57,58,59,60,61,62,63,64]
            ],
            [
                [1,2,3,4,5,6,7,8],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [57,58,59,60,61,62,63,64]
            ]
        ]
    """
    #print(sparse_values.view(2, 8, 8))

    #values, row_indices, row_offsets, column_indices = dense_to_sparse(a)

    """
        Dense Matrix: [
            [
                [1,2,3,4,5,6,7,8],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [57,58,59,60,61,62,63,64]
            ],
            [
                [65,66,67,68,69,70,71,72],
                [73..............],
                [81...............],
                [89...............],
                [97...............],
                [105...............],
                [113...............],
                [121,122,123,124,125,126,127,128]
            ]
        ]
    """
    dense = torch.arange(1, 2 * (nnz + 1) - 1).view(2, k, n).cuda().to(torch.float32)
    #print(dense)
    bias = torch.zeros((2, n)).cuda()

    result = torch_sputnik.replicated_spmm(2, m, k, n, nnz, sparse_row_indices, sparse_values, sparse_row_offsets, sparse_column_indices, dense, bias)

    """
        Output Matrix: [
            [
                [1380,1416,1452,1488,1524,1560,1596,1632],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [...............],
                [57,58,59,60,61,62,63,64]
            ],
            [
                [3684,3720,3756,3792,3828,3864,3900,3936],
                [9636..............],
                [15588...............],
                [21540...............],
                [27492...............],
                [33444...............],
                [39396...............],
                [45348,45832,46316,46800,47284,47768,48252,48736]
            ]
        ]
    """
    #print(result)

    return result

def dense_mm(m, k, n, nnz):
    a = torch.arange(1, (2 * nnz) + 1, dtype=torch.float32).view(2, m, k).cuda()
    #print(a.size())
    #print(a)
    dense = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    #print(dense.size())
    #print(dense)

    result = torch.bmm(a, dense)

    #print(result.size())
    #print(result)

    return result

if __name__ == "__main__":
    sparse_result = tensor_spmm(8, 8, 8, 64)
    dense_result = dense_mm(8, 8, 8, 64)

    if (dense_result != sparse_result).sum().item() == 0:
        print("Results are correct")