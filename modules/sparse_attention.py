import math
import torch
import torch.nn as nn
import torch_sputnik
import copy
import time
import numpy as np
from .sddmm import Sddmm
from .spmm import Spmm

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().clone().detach().to(torch.int32)
     row_indices = diffsort(row_offsets).to(torch.int32)
     column_indices = csr.col_indices().clone().detach().to(torch.int32)

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True)

def generate_mask(m, n, device, sparsity=0.9):
    num_elements = m * n

    num_zeros = int(num_elements * sparsity)
    num_ones = int(num_elements - num_zeros)

    mask = np.array([0] * num_zeros + [1] * num_ones)
    np.random.shuffle(mask)

    return torch.from_numpy(mask).reshape(m, n).cuda(device)

class SparseAttention(torch.nn.Module):
    def __init__(self, number_of_heads, model_dimension):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'
        
        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def attention(self, query, key, value, mask):
        q3d = self.four_d_to_three_d(query)
        k3d = self.four_d_to_three_d(key)
        v3d = self.four_d_to_three_d(value)
        mask2d = generate_mask(q3d.size(1), k3d.size(1), query.device, sparsity=0.9)
        _, row_indices, row_offsets, column_indices = dense_to_sparse(mask2d)

        m = q3d.size(1)
        n = k3d.size(1)

        scores = self.sddmm(
                    m, n,
                    row_indices, 
                    row_offsets, 
                    column_indices, 
                    q3d, 
                    k3d
                ) / math.sqrt(self.head_dimension)


        attention_weights = torch_sputnik.sparse_softmax(
                    scores, 
                    row_indices, 
                    row_offsets, 
                    column_indices
                )


        intermediate_token_representations = self.spmm(
                m, n,
                attention_weights,
                row_indices, 
                row_offsets, 
                column_indices, 
                v3d
            )

        return intermediate_token_representations

    def four_d_to_three_d(self, tensor):
        n, c, h, w = tensor.size()

        return tensor.reshape(n * c, h, w)
    
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        first, second, third, fourth = value.size(0), value.size(1), value.size(2), value.size(3)

        intermediate_token_representations = self.attention(query, key, value, mask)
        intermediate_token_representations = intermediate_token_representations.reshape(first, second, third, fourth)

        transposed = intermediate_token_representations.transpose(1, 2)

        reshaped = transposed.reshape(batch_size, -1, self.number_of_heads * self.head_dimension)
        
        token_representations = self.out_projection_net(reshaped)

        return token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])