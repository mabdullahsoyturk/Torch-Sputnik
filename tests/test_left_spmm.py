import torch
import torch.nn as nn
import torch_sputnik
import torch.nn.utils.prune as prune
from torch.autograd import gradcheck

torch.manual_seed(0)

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach().requires_grad_(True)
     row_offsets = csr.crow_indices().clone().detach().to(torch.int32)
     row_indices = diffsort(row_offsets)
     column_indices = csr.col_indices().clone().detach().to(torch.int32)

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.empty(output_features, input_features))

    def forward(self, x):
        values, row_indices, row_offsets, column_indices = dense_to_sparse(self.weight)
        return SparseLinearFunction.apply(self.weight.size(0),self.weight.size(1), values, row_indices, row_offsets, column_indices, x.transpose(1,2).contiguous())

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
                                        grad_output.contiguous(), 
                                        dense.contiguous())

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
                                        grad_output.contiguous())

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

def copy_params(linear, sparse_linear):
    sparse_linear.weight = nn.Parameter(linear.weight.detach().clone())

if __name__ == '__main__':
    batch_size = 32
    m, k, n = 1024, 768, 512
    
    linear = nn.Linear(k, m).cuda()
    linear.bias = nn.Parameter(torch.zeros_like(linear.bias))
    #prune.random_unstructured(linear, name="weight", amount=0.9)
    #prune.remove(linear, 'weight')

    sparse_linear = SparseLinear(k, m).cuda()

    x = torch.randn(batch_size, n, k).cuda()

    dense_output = linear(x)
    print(dense_output)
    print(f'dense_output size: {dense_output.size()}')

    copy_params(linear, sparse_linear)

    sparse_output = sparse_linear(x).transpose(1,2)
    print(sparse_output)
    print(f'sparse_output size: {sparse_output.size()}')

    if ((abs(dense_output) - abs(sparse_output)) < 1e-2).sum() == m * n * batch_size:
        print("Output matches")
    else:
        print("Doesn't match")

    dummy = torch.ones_like(sparse_output)
    loss = (dummy - sparse_output).sum()
    loss.backward()
    
    print(loss.grad)