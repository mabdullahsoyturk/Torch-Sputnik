import torch
import torch_sputnik
import numpy as np

class Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnz, row_indices, values, row_offsets, column_indices, b, c):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnz = nnz
        ctx.save_for_backward(values, row_indices, row_offsets, column_indices, b)
        return torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, b, c)

    @staticmethod
    def backward(ctx, grad_output):
        values, row_indices, row_offsets, column_indices, b, = ctx.saved_tensors
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnz = ctx.nnz
        out = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, grad_output, b, values)
        return None, None, None, None, None, out, None, None, None, None

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().clone().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().clone().to(torch.int32)

     return values.cuda().requires_grad_(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

device = torch.device("cuda:0")  # Uncomment this to run on GPU

nnz = 64
m = 8
k = 8
n = 8
x = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
values, row_indices, row_offsets, column_indices = dense_to_sparse(x)
y = torch.arange(1, nnz + 1, dtype=torch.float32, device=device, requires_grad=False).view(k, n)
z = torch.arange(1, nnz + 1, dtype=torch.float32, device=device, requires_grad=False).view(k, n)

correct_result = torch.arange(1, nnz + 1, dtype=torch.float32, device=device).view(k, n)

# To apply our Function, we use Function.apply method. We alias this as 'P3'.
P3 = Spmm.apply

# Forward pass: compute predicted y using operations; we compute
# P3 using our custom autograd operation.
y_pred = P3(m, k, n, nnz, row_indices, values, row_offsets, column_indices, y, z)

# Compute and print loss
print(y_pred - correct_result)
loss = (y_pred - correct_result).pow(2).sum()
print(loss.item())

# Use autograd to compute the backward pass.
print(f'Before: {values.grad}')
loss.backward()
print(f'After: {values.grad}')