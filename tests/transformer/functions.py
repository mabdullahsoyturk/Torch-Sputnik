import torch
import torch_sputnik
from utils import diffsort

class Spmm(torch.autograd.Function):
    """ Sparse Matrix Multiplication
    
    Takes one sparse and one dense matrix as inputs. Performs the matrix multiplication. 
    Sparse matrix is in CSR format. values, row indices, row_offsets and column indices 
    represent the sparse matrix.
    
    Operation: SparseMatrix x DenseMatrix = DenseOutput 
    
    Arguments:
        m: Number of rows in the sparse matrix.
        k: Number of columns in the sparse matrix.
        values: Nonzero elements in the sparse matrix.
        row_indices: Row indices of the output matrix. This is needed for load balance in CUDA kernel.
        row_offsets: Row offsets of the mask.
        column_indices: Column indices of the nonzero elements.
        dense: Dense matrix
    """

    @staticmethod
    def forward(ctx, b, m, k, nonzeros, values, row_indices, row_offsets, column_indices, dense):
        ctx.b = b
        ctx.m = m
        ctx.k = k
        ctx.nonzeros = nonzeros
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        #print(f'm: {m}, k: {k}, values: {values.size()}, row_indices: {row_indices.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, dense: {dense.size()}')
        
        result = torch_sputnik.spmm_many_mask(b, m, k, nonzeros, values, row_indices, row_offsets, column_indices, dense)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f'Spmm backward works')
        m = ctx.m
        k = ctx.k
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        #print(f'grad_output: {grad_output.size()}, dense: {dense.size()}')
        grad_values = torch_sputnik.sddmm(m, k,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)

        #print(f'[SpMM GRAD] values: {values.size()}')
        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                    values[0], 
                                    row_offsets, 
                                    column_indices)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.left_spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        print(f'Spmm backward finished')

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

class CsrSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, m, nonzeros, scores, row_indices, row_offsets, column_indices):
        ctx.b = b
        ctx.m = m
        ctx.nonzeros = nonzeros
        ctx.scores = scores
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices

        result = torch_sputnik.sparse_softmax_many_mask(
                    b, m, nonzeros,
                    scores, 
                    row_indices, 
                    row_offsets, 
                    column_indices)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f'Softmax backward works')
        scores = ctx.scores
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices

        grad_scores = grad_row_indices = grad_row_offsets = grad_column_indices = None

        I = torch.eye(grad_output.shape[0], grad_output.shape[1]).cuda()

        softmax = torch_sputnik.sparse_softmax(
                        grad_output, 
                        row_indices, 
                        row_offsets, 
                        column_indices)

        grad_scores = softmax * (I - softmax)

        print(f'Softmax backward finished')

        return grad_scores, grad_row_indices, grad_row_offsets, grad_column_indices

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
    def forward(ctx, b, m, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix):
        ctx.b = b
        ctx.m = m
        ctx.n = n
        ctx.nonzeros = nonzeros
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)

        result = torch_sputnik.sddmm_many_mask(b, m, n, nonzeros, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f'Sddmm backward works')
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

        #print(f'[SDDMM GRAD] values: {grad_output.size()}')
        grad_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, n, 
                                    grad_output[0], 
                                    row_offsets, 
                                    column_indices)

        row_indices_t = diffsort(row_offsets_t)

        # rhs grad
        grad_rhs = torch_sputnik.left_spmm(n, m,
                                    grad_t, 
                                    row_indices_t, 
                                    row_offsets_t, 
                                    column_indices_t, 
                                    lhs_matrix)

        print(f'Sddmm backward finished')

        return grad_m, grad_n, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs

class SparseLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, dense):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        result = torch_sputnik.left_spmm(m, k, values, row_indices, row_offsets, column_indices, dense)

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
                                        grad_output, 
                                        dense)

        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                                                                values, 
                                                                                row_offsets, 
                                                                                column_indices)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.left_spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense