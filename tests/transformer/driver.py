import torch
from modules import Transformer

def main():
    batch_size = 4
    N, seq_length, hidden_size, num_attention_heads, ffn_hidden_size = 6, 512, 512, 8, 2048
    batch = torch.rand((seq_length, batch_size, hidden_size)).cuda()
    mask = torch.rand((batch_size, 1, seq_length, seq_length)).cuda()

    mask[3][0][0][0] = 0
    mask[3][0][0][1] = 0
    mask[3][0][0][2] = 0
    mask[3][0][0][3] = 0

    transformer = Transformer(N, seq_length, hidden_size, num_attention_heads, ffn_hidden_size).cuda()
    output = transformer(batch, mask)

    print(output.size())

    torch.sum(output.flatten()).backward()

if __name__ == '__main__':
    main()