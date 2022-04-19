import torch
import torch_sputnik
import numpy as np

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnz, row_indices, values, row_offsets, column_indices, b, bias, c):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnz = nnz
        ctx.save_for_backward(values, row_indices, row_offsets, column_indices, b, bias)
        return torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, b, bias, c)

    @staticmethod
    def backward(ctx, grad_output):
        values, row_indices, row_offsets, column_indices, b, bias = ctx.saved_tensors
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnz = ctx.nnz
        sparse_matrix_grad = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, grad_output, b, values)

        values_t = values.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnz, values, row_offsets, column_indices, values_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)

        dense_matrix_grad = torch_sputnik.spmm(k, m, n, nnz, row_indices_t, values_t, row_offsets_t, column_indices_t, b, bias, grad_output)

        return None, None, None, None, None, sparse_matrix_grad, None, None, dense_matrix_grad, None, None

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
bias = torch.zeros((n), dtype=torch.float32, device=device)
z = torch.arange(1, nnz + 1, dtype=torch.float32, device=device, requires_grad=False).view(k, n)

correct_result = torch.arange(1, nnz + 1, dtype=torch.float32, device=device).view(k, n)

# To apply our Function, we use Function.apply method. We alias this as 'P3'.
P3 = Spmm.apply

# Forward pass: compute predicted y using operations; we compute
# P3 using our custom autograd operation.
y_pred = P3(m, k, n, nnz, row_indices, values, row_offsets, column_indices, y, bias, z)

# Compute and print loss
print(y_pred - correct_result)
loss = (y_pred - correct_result).pow(2).sum()
print(loss.item())

# Use autograd to compute the backward pass.
print(f'Before: {values.grad}')
print(f'Before: {y.grad}')
loss.backward()
print(f'After: {values.grad}')
print(f'After: {y.grad}')