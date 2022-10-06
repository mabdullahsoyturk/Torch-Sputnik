import argparse

import numpy as np
import torch

def generate_mask(m, n, device, sparsity=0.9, round_to=4):
    num_elements = m * n

    remainder = int(num_elements * sparsity) % 4

    num_zeros = int(num_elements * sparsity) - remainder
    num_ones = int(num_elements - num_zeros)

    mask = np.array([0] * num_zeros + [1] * num_ones)
    np.random.shuffle(mask)

    return torch.from_numpy(mask).reshape(m, n).cuda(device)

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
    row_indices = torch.stack(row_indices_list)
    row_offsets = torch.stack(row_offsets_list)
    column_indices = torch.cat(column_indices_list)
    nnzs = torch.tensor(nnzs)

    return values, row_indices, row_offsets, column_indices, nnzs

def dense_to_sparse(matrix):
    csr = matrix.to_sparse_csr()
    values = csr.values().detach().to(torch.float32).requires_grad_(True)
    row_offsets = csr.crow_indices().to(torch.int32)
    row_indices = diffsort(row_offsets)
    column_indices = csr.col_indices().to(torch.int32)

    return values, row_indices, row_offsets, column_indices, csr._nnz()

def diffsort(offsets):
    diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
    return torch.argsort(diffs, descending=True).to(torch.int32)

def diffsort_many_mask(offsets):
    num_masks = offsets.size(0)

    row_indices_list = []

    for idx in range(num_masks):
        indices = diffsort(offsets[idx])
        row_indices_list.append(indices)

    return torch.stack(row_indices_list)

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(tensor, num_partitions):
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    # Note: torch.split does not create contiguous tensors by default.
    return tuple(chunk.contiguous() for chunk in tensor_list)