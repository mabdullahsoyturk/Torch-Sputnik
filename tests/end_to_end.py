import torch
import torch.nn as nn
import torch_sputnik

class SparseCoreAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        
        self.hidden_size_per_attention_head = hidden_size / num_attention_heads

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

class SparseAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = hidden_size / num_attention_heads

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, self.args.hidden_size)

        self.core_attention = SparseCoreAttention(hidden_size, num_attention_heads)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [s, b, h] --> [b, s, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()

        # mixed_x_layer: [b, s, (n * hn * 3)]
        mixed_x_layer = self.query_key_value(hidden_states).transpose(1, 2).contiguous()

        # new_tensor_shape: [b, s, n, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_size_per_attention_head)

        # mixed_x_layer: [b, sq, np, 3 * hn]
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape).contiguous()
        pass

class MLP(torch.nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()

        self.to_4h = nn.Linear(hidden_size, ffn_hidden_size)
        self.to_h  = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, hidden_states):
        # hidden_states = [s, b, h] -> [512, 16, 1024]
        pass

class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = SparseAttention()
        self.mlp = MLP()

    def forward(self, hidden_states, attention_mask):
        # hidden_states = [s, b, h] -> [512, 16, 1024], attention_mask = [b, 1, s, s] -> [16, 1, 512, 512]
        attention_output = self.self_attention(hidden_states, attention_mask)

        mlp_output = self.mlp(attention_output)

        # output [s, b, h] -> [512, 16, 1024]
        return mlp_output