import torch
import torch_sputnik

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().data.clone().to(torch.int32)
     row_indices = diffsort(row_offsets)
     column_indices = csr.col_indices().data.clone().to(torch.int32)

     return values, row_indices, row_offsets, column_indices, torch.Tensor([values.size(-1)]).to(torch.int32).cpu()

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True)

def dense_to_sparse_3d(dense):
    replication = dense.size(0)

    values_3d, row_indices_3d, row_offsets_3d, column_indices_3d, nnz_3d = dense_to_sparse(dense[0, :, :])

    for idx in range(1, replication):
        values, row_indices, row_offsets, column_indices, nnz = dense_to_sparse(dense[idx, :, :])

        values_3d = torch.cat([values_3d, values])
        row_indices_3d = torch.cat([row_indices_3d, row_indices])
        row_offsets_3d = torch.cat([row_offsets_3d, row_offsets])
        column_indices_3d = torch.cat([column_indices_3d, column_indices])
        nnz_3d = torch.cat([nnz_3d, nnz])

    return values_3d.cuda(), row_indices_3d.type(torch.IntTensor).cuda(), row_offsets_3d.type(torch.IntTensor).cuda(), column_indices_3d.type(torch.IntTensor).cuda(), nnz_3d

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

    print(sparse_result == dense_result)