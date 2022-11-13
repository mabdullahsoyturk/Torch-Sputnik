import torch
import torch_sputnik
from utils import diffsort, diffsort_many_mask

class Spmm(torch.autograd.Function):

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
        #print(f'Spmm backward works')
        b = ctx.b
        m = ctx.m
        k = ctx.k
        nonzeros = ctx.nonzeros
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_b = grad_m = grad_k = grad_nonzeros = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_dense = None

        # sparse matrix grad
        #print(f'[SpMM GRAD] grad_output: {grad_output.size()}, dense: {dense.size()}')
        # grad_output: [32, 512, 64], row_indices: [2048], row_offsets: [2052], column_indices: [525312], dense: [32, 512, 64]
        grad_values = torch_sputnik.sddmm_many_mask(b, m, k, nonzeros,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        grad_output, 
                                        dense)

        #print(f'[SpMM GRAD] csr_transpose --> m: {m}, k: {k}, values: {values.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        # m: 512, k: 512, values: [32, 131328]
        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose_many_mask(b, m, k, nonzeros, 
                                    values,
                                    row_offsets,
                                    column_indices)
        
        row_indices_t = diffsort_many_mask(row_offsets_t)

        # dense matrix grad
        #print(f'[SpMM GRAD] spmm --> k: {k}, m: {k}, nonzeros: {nonzeros}, values_t: {values_t.size()}, row_indices_t: {row_indices_t.size()}, row_offsets_t: {row_offsets_t.size()}, column_indices_t: {column_indices_t.size()}, grad_output: {grad_output.size()}')
        grad_dense = torch_sputnik.spmm_many_mask(b, k, m, nonzeros, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output)

        #print(f'Spmm backward finished')

        return grad_b, grad_m, grad_k, grad_nonzeros, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

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
        #print(f'Softmax backward works')
        b = ctx.b
        m = ctx.m
        nonzeros = ctx.nonzeros
        scores = ctx.scores
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices

        grad_b = grad_m = grad_nonzeros = grad_scores = grad_row_indices = grad_row_offsets = grad_column_indices = None

        #I = torch.eye(grad_output.shape[0], grad_output.shape[1]).cuda()

        softmax = torch.nn.functional.softmax(grad_output, dim=1)
        #softmax = torch_sputnik.sparse_softmax_many_mask(
        #                b, m, nonzeros,
        #                grad_output, 
        #                row_indices, 
        #                row_offsets, 
        #                column_indices)
        print(f'grad_output: {grad_output}, softmax_result: {softmax}')

        #print(f'grad_output: {grad_output.size()}, softmax_out: {softmax.size()}')
        grad_scores = softmax * (1 - softmax)
        #print(f'grad_scores: {grad_scores}')

        #print(f'Softmax backward finished')

        return grad_b, grad_m, grad_nonzeros, grad_scores, grad_row_indices, grad_row_offsets, grad_column_indices

class Sddmm(torch.autograd.Function):

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
        #print(f'Sddmm backward works')
        b = ctx.b
        m = ctx.m
        n = ctx.n
        nonzeros = ctx.nonzeros
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_b = grad_m = grad_n = grad_nonzeros = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = None
        
        # lhs grad
        #print(f'[SDDMM GRAD] b: {b}, m: {m}, n: {n}, nonzeros: {nonzeros}, grad_output: {grad_output.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}, rhs_matrix: {rhs_matrix.size()}')
        # b: 4, m: 512, n: 512, nonzeros: [4], grad_output: [32, 131328], row_offsets: [2052], column_indices: [525312], rhs_matrix: [32, 512, 64]
        grad_lhs = torch_sputnik.spmm_many_mask(b, m, n, nonzeros, 
                                    grad_output,
                                    row_indices, 
                                    row_offsets, 
                                    column_indices, 
                                    rhs_matrix)

        #print(f'[SDDMM GRAD] grad_output: {grad_output.size()}, row_offsets: {row_offsets.size()}, column_indices: {column_indices.size()}')
        # grad_output: [32, 131328], row_offsets: [2052], column_indices: [525312]
        grad_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose_many_mask(b, m, n, nonzeros, 
                                    grad_output, 
                                    row_offsets, 
                                    column_indices)

        #print(f'[SDDMM GRAD] grad_t: {grad_t.size()}, row_offsets_t: {row_offsets_t.size()}, column_indices_t: {column_indices_t.size()}')
        # grad_output_t: [32, 131328], row_offsets_t: [2052], column_indices_t: [525312]
        row_indices_t = diffsort_many_mask(row_offsets_t)

        # rhs grad
        #print(f'[SDDMM GRAD] grad_t: {grad_t.size()}, lhs_matrix: {lhs_matrix.size()}')
        # grad_output_t: [32, 131328], row_offsets_t: [2052], column_indices_t: [525312]
        grad_rhs = torch_sputnik.spmm_many_mask(b, n, m, nonzeros,
                                    grad_t, 
                                    row_indices_t, 
                                    row_offsets_t, 
                                    column_indices_t, 
                                    lhs_matrix)

        #print(f'Sddmm backward finished')

        return grad_b, grad_m, grad_n, grad_nonzeros, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs

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
