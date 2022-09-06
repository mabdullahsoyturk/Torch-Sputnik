import math
import torch
import torch.nn as nn
import torch_sputnik
import copy
import time
import numpy as np
from .sddmm import Sddmm
from .spmm import Spmm
from .sparse_linear import SparseLinear

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

def generate_mask(m, n, device, sparsity=0.9, round_to=4):
    num_elements = m * n

    remainder = int(num_elements * sparsity) % 4

    num_zeros = int(num_elements * sparsity) - remainder
    num_ones = int(num_elements - num_zeros)

    mask = np.array([0] * num_zeros + [1] * num_ones)
    np.random.shuffle(mask)

    return torch.from_numpy(mask).reshape(m, n).cuda(device)

class SparseAttention(torch.nn.Module):
    def __init__(self, num_heads, embedding_size, max_sequence_length=512):
        super().__init__()
        assert embedding_size % num_heads == 0, f'Model dimension must be divisible by the number of heads.'
        
        self.head_dim = embedding_size // num_heads
        self.num_heads = num_heads

        #self.linears = get_clones(nn.Linear(embedding_size, embedding_size, bias=False), 4)
        self.linears = get_clones(SparseLinear(embedding_size, embedding_size), 4)

        self.m = max_sequence_length
        self.n = max_sequence_length
        self.mask2d = generate_mask(self.m, self.n, 'cuda', sparsity=0.9)
        _, self.row_indices, self.row_offsets, self.column_indices = dense_to_sparse(self.mask2d)

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def attention(self, query, key, value, mask):
        q3d = self.four_d_to_three_d(query)
        k3d = self.four_d_to_three_d(key)
        v3d = self.four_d_to_three_d(value)

        # IN = q3d: (256, 512, 96)
        # IN = k3d: (256, 512, 96)
        # OUT = scores: (256, 262144)
        scores = self.sddmm(
                    self.m, self.n,
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices, 
                    q3d, 
                    k3d
                ) / math.sqrt(self.head_dim)

        #print(f'scores: {scores.size()}')

        attention_weights = torch_sputnik.sparse_softmax(
                    scores, 
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices
                )

        #print(f'attention_weights: {attention_weights.size()}')
        # IN = attention_weights: (256, 262144)
        # IN = v3d: (256, 512, 96)
        # OUT = representations: (256, 512, 96)
        intermediate_token_representations = self.spmm(
                self.m, self.n,
                attention_weights,
                self.row_indices, 
                self.row_offsets, 
                self.column_indices, 
                v3d
            )

        #print(f'\nq3d: {q3d.size()}, k3d: {k3d.size()}, v3d: {v3d.size()}, mask2d: {mask2d.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, m: {m}, k: {k}, n: {n}, scores: {scores.size()}, attention_weights: {attention_weights.size()}, intermediate_token_representations: {intermediate_token_representations.size()}')

        return intermediate_token_representations

    def four_d_to_three_d(self, tensor):
        n, c, h, w = tensor.size()

        return tensor.reshape(n * c, h, w)
    
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [net(x).transpose(1,2).contiguous()
                                    .view(batch_size, -1, self.num_heads, self.head_dim)
                                    .transpose(1, 2)
                            for net, x in zip(self.linears, (query, key, value))]
        #print(f'batch_size: {batch_size}, query: {query.size()}, key: {key.size()}, value: {value.size()}, mask: {mask.size()}')

        first, second, third, fourth = value.size(0), value.size(1), value.size(2), value.size(3)

        intermediate_token_representations = self.attention(query, key, value, mask)
        intermediate_token_representations = intermediate_token_representations.reshape(first, second, third, fourth)
        #print(f'Representations reshaped to: {first, second, third, fourth}')

        transposed = intermediate_token_representations.transpose(1, 2).contiguous()
        #print(f'Representations transposed to: {transposed.size()}')

        reshaped = transposed.reshape(batch_size, -1, self.num_heads * self.head_dim)
        #print(f'Representations transposed, then reshaped to: {reshaped.size()}')
        
        token_representations = self.linears[-1](reshaped).transpose(1,2)

        return token_representations

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])