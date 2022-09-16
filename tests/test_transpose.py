import torch
import torch_sputnik

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().data.clone().to(torch.int32)
     row_indices = diffsort(row_offsets).to(torch.int32)
     column_indices = csr.col_indices().data.clone().to(torch.int32)

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

def sparse_transpose(sparse, m, k, n):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(sparse)

    output_values, output_row_offsets, output_column_indices = torch_sputnik.csr_transpose(m, n, values, row_offsets, column_indices)

    return output_values

if __name__ == "__main__":
    m, k, n = 64, 64, 64
    sparse = torch.rand((m * k), dtype=torch.float32).view(m, k).cuda()

    sparse_result = sparse_transpose(sparse, m, k, n).reshape(k, m)

    print(sparse_result)
    print(sparse.t())

    if ((sparse_result - sparse.t()) < 1e-4).sum().item() == m * n:
        print("Results match")
    else:
        print("Results don't match")
