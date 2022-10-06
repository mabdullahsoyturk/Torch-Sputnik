import math

import torch
import torch.nn as nn
import torch_sputnik

from utils import *
from functions import Sddmm, CsrSoftmax, Spmm, SparseLinearFunction

class SparseCoreAttention(torch.nn.Module):
    
    def __init__(self, seq_length, hidden_size, num_attention_heads):
        super().__init__()
        
        self.seq_length = seq_length
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.mask2d = generate_mask(m=512, n=512, device=0, sparsity=0.9)
        _, self.row_indices, self.row_offsets, self.column_indices, _ = dense_to_sparse(self.mask2d)

        self.sddmm = Sddmm.apply
        self.softmax = CsrSoftmax.apply
        self.spmm = Spmm.apply

    def four_d_to_three_d(self, tensor):
        b, n, s, hn = tensor.size()

        return tensor.reshape(b * n, s, hn)

    def forward(self, query, key, value, mask):
        # query, key, value: each [b, s, n, hn]
        b = mask.size(0)
        values, row_indices, row_offsets, column_indices, nnzs = dense_to_sparse_3d(mask.squeeze())

        # output_shape: [s, b, h]
        output_shape = (query.size(1), query.size(0), query.size(2) * query.size(3))

        # Input query_layer, key_layer, value_layer: each [b, s, n, hn] --> [b, n, s, hn]
        query = torch.permute(query, (0, 2, 1, 3))
        key = torch.permute(key, (0, 2, 1, 3))
        value = torch.permute(value, (0, 2, 1, 3))

        # Query, key, value: each [b, n, s, hn] --> [b * n, s, hn]
        query = self.four_d_to_three_d(query)
        key = self.four_d_to_three_d(key)
        value = self.four_d_to_three_d(value)

        #print(f'row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, query: {query.size()}, key: {key.size()}')
        # row_indices: [2048], row_offsets: [2052], column_indices: [525312], query: [32, 512, 64], key: [32, 512, 64]
        scores = self.sddmm(b,
                    self.seq_length, self.seq_length,
                    nnzs,
                    row_indices, 
                    row_offsets, 
                    column_indices, 
                    query, 
                    key
        ) / math.sqrt(self.hidden_size_per_attention_head)

        #print(f'scores: {scores.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        # scores: [32, 131328]
        weights = self.softmax(
                    b, self.seq_length, nnzs,
                    scores, 
                    row_indices, 
                    row_offsets, 
                    column_indices
        )

        #print(f'weights: {weights.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, value: {value.size()}')
        # weights: [32, 131328], value: [32, 512, 64]
        representations = self.spmm(b,
                self.seq_length, self.seq_length,
                nnzs,
                weights,
                row_indices, 
                row_offsets, 
                column_indices, 
                value
        )

        # representations: [s, b, h]
        representations = torch.permute(representations, (1, 0, 2)).reshape(*output_shape)

        return representations

class SparseAttention(torch.nn.Module):
    
    def __init__(self, seq_length, hidden_size, num_attention_heads):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.core_attention = SparseCoreAttention(seq_length, hidden_size, num_attention_heads)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [s, b, h] --> [b, s, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()

        # mixed_x_layer: [b, s, (n * hn * 3)]
        mixed_x_layer = self.query_key_value(hidden_states).contiguous()

        # new_tensor_shape: [b, s, n, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_size_per_attention_head)

        # mixed_x_layer: [b, s, n, 3 * hn]
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape).contiguous()

        # query, key, value: each [b, s, n, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # context_layer: [s, b, h]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # output: [s, b, h]
        output = self.dense(context_layer).contiguous()

        return output

class MLP(torch.nn.Module):
    
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()

        self.to_4h = nn.Linear(hidden_size, ffn_hidden_size)
        self.to_h  = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, hidden_states):
        # hidden_states = [s, b, h] -> [512, 16, 1024]
        output_shape = hidden_states.size()

        # hidden_states = [b, s, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()

        # hidden_states = [b, s, 4h]
        hidden_states = self.to_4h(hidden_states)

        # hidden_states = [b, s, h]
        hidden_states = self.to_h(hidden_states)

        # output = [s, b, h]
        output = hidden_states.reshape(*output_shape)
        
        return output

class TransformerLayer(torch.nn.Module):
    
    def __init__(self, seq_length, hidden_size, num_attention_heads, ffn_hidden_size):
        super().__init__()
        self.self_attention = SparseAttention(seq_length, hidden_size, num_attention_heads)
        self.mlp = MLP(hidden_size, ffn_hidden_size)

    def forward(self, hidden_states, attention_mask):
        # hidden_states = [s, b, h] -> [512, 16, 1024], attention_mask = [b, 1, s, s] -> [16, 1, 512, 512]
        attention_output = self.self_attention(hidden_states, attention_mask)

        mlp_output = self.mlp(attention_output)

        # output [s, b, h] -> [512, 16, 1024]
        return mlp_output

class Transformer(torch.nn.Module):
    def __init__(self, N, seq_length, hidden_size, num_attention_heads, ffn_hidden_size):
        super().__init__()

        self.N = N
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size

        self.layers = [TransformerLayer(seq_length, hidden_size, num_attention_heads, ffn_hidden_size) for _ in range(N)]
        self.init_modules()

    def init_modules(self):
        for index, layer in enumerate(self.layers):
            self.add_module(f'{index}', layer)

    def forward(self, hidden_states, mask):
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        
        return hidden_states
