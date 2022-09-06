import torch
import torch_sputnik

def diffsort(offsets):
    """ Extracts row indices from given row offsets"""
    diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
    return torch.argsort(diffs, descending=True).to(torch.int32)

class Sddmm(torch.autograd.Function):
    """ Sampled Dense Dense Matrix Multiplication
    
    Takes two dense matrices as inputs. Performs the matrix multiplication and samples
    the entries specified by the mask. Mask is in CSR format. Row indices, row_offsets
    and column indices represent the mask.
    
    Operation: (LHS x RHS) . mask = Output 
    
    Arguments:
        m: Number of rows in left hand side (lhs) matrix.
        n: Number of columns in right hand side (rhs) matrix.
        row_indices: Row indices of the output matrix. This is needed for load balance in CUDA kernel.
        row_offsets: Row offsets of the mask.
        column_indices: Column indices of the nonzero elements.
        lhs_matrix: Left Hand Side matrix
        rhs_matrix: Right Hand Side matrix
    """  
    
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

        grad_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, n, 
                                    grad_output, 
                                    row_offsets, 
                                    column_indices)

        row_indices_t = diffsort(row_offsets_t)

        # rhs grad
        grad_rhs = torch_sputnik.spmm(n, m,
                                    grad_t, 
                                    row_indices_t, 
                                    row_offsets_t, 
                                    column_indices_t, 
                                    lhs_matrix)

        return grad_m, grad_n, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs