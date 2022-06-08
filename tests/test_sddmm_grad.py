import numpy as np
import torch
import torch_sputnik
from torch.autograd import gradcheck

import connectors
import initializers
import sparse_matrix

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().data.clone().to(torch.int32)
     row_indices = diffsort(row_offsets).to(torch.int32)
     column_indices = csr.col_indices().data.clone().to(torch.int32)

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class Sddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix):
        ctx.m = m
        ctx.n = n
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)

        result = torch_sputnik.sddmm(m, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        n = ctx.n
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_m = grad_n = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = None
        
        # lhs grad
        grad_lhs = torch_sputnik.spmm(m, n, 
                                    grad_output,
                                    row_indices, 
                                    row_offsets, 
                                    column_indices, 
                                    rhs_matrix)

        grad_t = torch.zeros_like(grad_output).t()
        row_offsets_t = torch.zeros_like(row_offsets).t()
        column_indices_t = torch.zeros_like(column_indices).t()

        torch_sputnik.csr_transpose(m, n, 
                                    grad_output, 
                                    row_offsets, 
                                    column_indices, 
                                    grad_t, 
                                    row_offsets_t, 
                                    column_indices_t)

        row_indices_t = diffsort(row_offsets_t)

        # rhs grad
        grad_rhs = torch_sputnik.spmm(n, m,
                                    grad_t, 
                                    row_indices_t, 
                                    row_offsets_t, 
                                    column_indices_t, 
                                    lhs_matrix)

        return grad_m, grad_n, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs


if __name__ == '__main__':
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0

    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([m, k])
    rhs_np = initializer([n, k])
    output_np = connector(np.ones([m, n]))

    output_topology = sparse_matrix.SparseMatrix(matrix=output_np)
    lhs = torch.from_numpy(lhs_np).to(torch.float32).requires_grad_().cuda()
    rhs = torch.from_numpy(rhs_np).to(torch.float32).requires_grad_().cuda()

    sddmm = Sddmm.apply

    output = sddmm(m, n, output_topology.row_indices, output_topology.row_offsets, output_topology.column_indices, lhs, rhs)