import math

import numpy as np

import torch
import torch_sputnik

def mm(lhs_matrix, rhs_matrix, masks):
    result = torch.matmul(lhs_matrix, rhs_matrix.transpose(-2, -1))
    result.masked_fill_(masks, 0)

    return result

def generate_mask(m, n, device, sparsity=0.9, round_to=4):
    num_elements = m * n

    remainder = int(num_elements * sparsity) % 4

    num_zeros = int(num_elements * sparsity) - remainder
    num_ones = int(num_elements - num_zeros)

    mask = np.array([0] * num_zeros + [1] * num_ones)
    np.random.shuffle(mask)

    return torch.from_numpy(mask).reshape(m, n).cuda(device)

def generate_mask_3d(b, m, n, device=0, round_to=4):
    sparsities = [0.2, 0.5]

    masks = []

    for index in range(b):
        mask = generate_mask(m, n, device, sparsities[index % 2], round_to)
        masks.append(mask)

    return torch.stack(masks)

def four_d_to_three_d(tensor):
    b, n, s, hn = tensor.size()

    return tensor.reshape(b * n, s, hn)

def diffsort(offsets):
    diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
    return torch.argsort(diffs, descending=True).to(torch.int32)

def dense_to_sparse(matrix):
    csr = matrix.to_sparse_csr()
    values = csr.values().detach().to(torch.float32).requires_grad_(True)
    row_offsets = csr.crow_indices().to(torch.int32)
    row_indices = diffsort(row_offsets)
    column_indices = csr.col_indices().to(torch.int32)

    return values, row_indices, row_offsets, column_indices, csr._nnz()

def dense_to_sparse_3d(mask):
    values_list = []
    row_indices_list = []
    row_offsets_list = []
    column_indices_list = []
    nnzs = []

    for index in range(mask.size(0)):
        values, row_indices, row_offsets, column_indices, nnz = dense_to_sparse(mask[index, :, :])
        values_list.append(values)
        row_indices_list.append(row_indices)
        row_offsets_list.append(row_offsets)
        column_indices_list.append(column_indices)
        nnzs.append(nnz)

    values = torch.cat(values_list)
    row_indices = torch.cat(row_indices_list)
    row_offsets = torch.cat(row_offsets_list)
    column_indices = torch.cat(column_indices_list)
    nnzs = torch.tensor(nnzs)

    return values, row_indices, row_offsets, column_indices, nnzs

### Dense
# [b, n, s, s]
# scores: [16, 16, 512, 512], mask: [16, 1, 512, 512]

### Sparse
# query: [16, 16, 512, 64]
# query_3d: [256, 512, 64], 
# scores: [256, 262144], 
# weights: [256, 262144], 
# mask: torch.Size([16, 1, 512, 512])

if __name__ == "__main__":
    b, n, s, hn = 16, 16, 512, 64

    # Init mask
    mask = generate_mask_3d(b, s, s, 0)

    # Init 4d query and key
    query = torch.rand((b, n, s, hn)).cuda()
    key = torch.rand((b, n, s, hn)).cuda()
    value = torch.rand((b, n, s, hn)).cuda()

    print(query.size())

    # Save output shape
    output_shape = (query.size(2),
                    query.size(0),
                    query.size(1) * query.size(3))

    # Reshape to 3d
    query = four_d_to_three_d(query)
    key = four_d_to_three_d(key)
    value = four_d_to_three_d(value)

    # Get values, row_indices, row_offsets, column_indices and nnzs
    values, row_indices, row_offsets, column_indices, nonzeros = dense_to_sparse_3d(mask)
    print(f'Mask size: {mask.size()}')
    print(f'values:{values.size()}, row_indices:{row_indices.size()}, row_offsets:{row_offsets.size()}, '
            f'column_indices:{column_indices.size()}, nnzs:{nonzeros.size()}')

    # SDDMM
    scores = torch_sputnik.sddmm_many_mask(
                b, s, s, nonzeros,
                row_indices, 
                row_offsets, 
                column_indices, 
                query, 
                key
    ) / math.sqrt(hn)

    print(scores.size())

    # Softmax
    weights = torch_sputnik.sparse_softmax_many_mask(
                b, s, nonzeros,
                scores, 
                row_indices, 
                row_offsets, 
                column_indices
    )

    print(weights.size())

    # SpMM
    representations = torch_sputnik.spmm_many_mask(
            b, s, s, nonzeros,
            weights,
            row_indices, 
            row_offsets, 
            column_indices, 
            value
    )

    print(representations.size())

    representations = torch.permute(representations, (1, 0, 2)).reshape(*output_shape)

    print(representations.size())