import torch
import torch_sputnik

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

        grad_m = grad_k = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        grad_values = torch_sputnik.sddmm(m, k,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output.contiguous(), 
                                        dense.contiguous())

        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                                                                values, 
                                                                                row_offsets, 
                                                                                column_indices)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output.contiguous())

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense