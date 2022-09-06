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

class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        self.bias = nn.Parameter(torch.empty(output_features))
        
    def setup_sparse_tensors(self):
        values, row_indices, row_offsets, column_indices = dense_to_sparse(self.weight)
        self.values = nn.Parameter(values)
        self.row_indices = row_indices
        self.row_offsets = row_offsets
        self.column_indices = column_indices

    def forward(self, x):
        #print(f'Sparsity of the linear layer: {(self.weight == 0).sum() / self.weight.numel()}')
        #print(f'X size: {x.size()}, W size: {self.weight.size()}, values: {self.values.size()}')
        #print(f'{type(self.weight.size(0))},{type(self.weight.size(1))}, {type(self.values)}, {type(self.row_indices)}, {type(self.row_offsets)}, {type(self.column_indices)}')
        return SparseLinearFunction.apply(self.output_features, self.input_features, self.values, self.row_indices, self.row_offsets, self.column_indices, x.transpose(1,2).contiguous())

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, dense):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        result = torch_sputnik.left_spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        grad_values = torch_sputnik.sddmm(m, k,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)

        #print(f'[SparseLinearFunction] values: {values.size()}')
        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                                                                values, 
                                                                                row_offsets, 
                                                                                column_indices)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.left_spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

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

    @nvtx.annotate("sparse_attention", color="red")
    def attention(self, query, key, value, mask):
        with nvtx.annotate("4d_3d"):
            q3d = self.four_d_to_three_d(query)
            k3d = self.four_d_to_three_d(key)
            v3d = self.four_d_to_three_d(value)
        #mask2d = torch.ones((q3d.size(1), k3d.size(1))).cuda(query.device).to(torch.float32)
        #mask2d = generate_mask(q3d.size(1), k3d.size(1), query.device, sparsity=0.9)
        #_, row_indices, row_offsets, column_indices = dense_to_sparse(mask2d)

        #print(f'\nq3d: {q3d.size()}, k3d: {k3d.size()}, v3d: {v3d.size()}, mask3d: {mask2d.size()}')
        
        #print(f'row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        
        #print(f'm: {m}, k: {k}, n: {n}')

        #start = time.time()
        # IN = q3d: (256, 512, 96)
        # IN = k3d: (256, 512, 96)
        # OUT = scores: (256, 262144)
        with nvtx.annotate("SDDMM"):
            scores = self.sddmm(
                        self.m, self.n,
                        self.row_indices, 
                        self.row_offsets, 
                        self.column_indices, 
                        q3d, 
                        k3d
                    ) / math.sqrt(self.head_dim)

        #print(f'scores: {scores.size()}')

        with nvtx.annotate("Sparse Softmax"):
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
        with nvtx.annotate("SPMM"):
            intermediate_token_representations = self.spmm(
                    self.m, self.n,
                    attention_weights,
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices, 
                    v3d
                )
        #end = time.time()
        #print(f'Sparse attention:{end - start}')

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