import torch
import torch.nn as nn
import torch_sputnik

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach()
     row_offsets = csr.crow_indices().clone().detach().to(torch.int32)
     row_indices = diffsort(row_offsets).to(torch.int32)
     column_indices = csr.col_indices().clone().detach().to(torch.int32)

     return values, row_indices, row_offsets, column_indices
 
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

        #print(f'[SparseLinearFunction] values: {values.size()}')
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

class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        self.bias = nn.Parameter(torch.empty(output_features))
        
    def setup_sparse_tensors(self):
        values, row_indices, row_offsets, column_indices = dense_to_sparse(self.weight)
        self.values = nn.Parameter(values)
        self.row_indices = row_indices
        self.row_offsets = row_offsets
        self.column_indices = column_indices

    def forward(self, x):
        #print(f'Sparsity of the linear layer: {(self.weight == 0).sum() / self.weight.numel()}')
        #print(f'X size: {x.size()}, W size: {self.weight.size()}, values: {self.values.size()}')
        #print(f'{type(self.weight.size(0))},{type(self.weight.size(1))}, {type(self.values)}, {type(self.row_indices)}, {type(self.row_offsets)}, {type(self.column_indices)}')
        return SparseLinearFunction.apply(self.output_features, self.input_features, self.values, self.row_indices, self.row_offsets, self.column_indices, x.transpose(1,2).contiguous())