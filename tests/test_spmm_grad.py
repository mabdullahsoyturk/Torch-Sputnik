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

class Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, dense):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)
        
        result = torch_sputnik.spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_row_indices = grad_values = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        grad_values = torch_sputnik.sddmm(m, k, 
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)

        values_t = torch.zeros_like(values)
        row_offsets_t = torch.zeros_like(row_offsets)
        column_indices_t = torch.zeros_like(column_indices)

        torch_sputnik.csr_transpose(m, k, 
                                    values, 
                                    row_offsets, 
                                    column_indices, 
                                    values_t, 
                                    row_offsets_t, 
                                    column_indices_t)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        return grad_m, grad_k, grad_nnz, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

if __name__ == '__main__':
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0

    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = connector(initializer([m, k]))
    rhs_np = initializer([k, n])

    lhs = sparse_matrix.SparseMatrix(matrix=lhs_np)
    rhs = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    spmm = Spmm.apply

    output = spmm(m, k, lhs.values, lhs.row_indices, lhs.row_offsets, lhs.column_indices, rhs)