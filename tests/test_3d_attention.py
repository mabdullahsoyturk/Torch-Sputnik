import numpy as np
import torch
import torch_sputnik
from torch.autograd import gradcheck

import connectors
import initializers
import sparse_matrix

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

        grad_t = torch.zeros_like(grad_output)
        row_offsets_t = torch.zeros_like(row_offsets)
        column_indices_t = torch.zeros_like(column_indices)

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

class SparseAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def forward(self, m, k, n, row_indices, row_offsets, column_indices, q3d, k3d, v3d):
        logits = self.sddmm(
                    m, n,
                    row_indices, 
                    row_offsets, 
                    column_indices, 
                    q3d, 
                    k3d,
                )

        weights = torch_sputnik.sparse_softmax(
                    logits, 
                    row_indices, 
                    row_offsets, 
                    column_indices
                )

        out = self.spmm(
                m, k,
                weights,
                row_indices,  
                row_offsets, 
                column_indices, 
                v3d
            )

        return out

if __name__ == "__main__":
    r, m, k, n, sparsity = 8, 512, 512, 512, 0.0

    connector = connectors.Uniform(sparsity)
    initializer = initializers.Uniform()

    # Numpy matrices for verification.
    lhs_np = initializer([m, k])
    rhs_np = initializer([m, k])
    output_np = connector(np.ones([m, n]))

    output_topology = sparse_matrix.SparseMatrix(matrix=output_np)
    q3d = torch.from_numpy(lhs_np).to(torch.float32).cuda()
    k3d = torch.from_numpy(rhs_np).to(torch.float32).cuda()
    v3d = torch.from_numpy(rhs_np).to(torch.float32).cuda()

    sparse_attention = SparseAttention()

    result = sparse_attention(m, k, n, output_topology.row_indices, output_topology.row_offsets, output_topology.column_indices, q3d, k3d, v3d)

    print(result)