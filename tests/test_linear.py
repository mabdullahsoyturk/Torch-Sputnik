import torch
import torch.nn as nn
import torch_sputnik
import torch.nn.utils.prune as prune

torch.manual_seed(0)

def dense_to_sparse(matrix):
     csr = matrix.to_sparse_csr()
     values = csr.values().clone().detach().requires_grad_(True)
     row_offsets = csr.crow_indices().clone().detach().to(torch.int32)
     row_indices = diffsort(row_offsets)
     column_indices = csr.col_indices().clone().detach().to(torch.int32)
     
     print(f'Dense to sparse -> values:{values.size()}, row_offsets: {row_offsets.size()}, row_indices: {row_indices.size()}, column_indices: {column_indices.size()}')

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class SparseLinear(nn.Module):
    # (w.xT).T
    # w = (256, 128), x = (72, 128)
    
    def __init__(self, input_features, output_features, bias=False):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, x):
        return SparseLinearFunction.apply(self.output_features, self.input_features, self.values, self.row_indices, self.row_offsets, self.column_indices, x)

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, dense):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        # x = (72, 128), w = (256, 128), op=(w.xT)T y = (72, 256)
        result = torch_sputnik.spmm(m, k, values, row_indices, row_offsets, column_indices, dense.t()).t()

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

        # sparse matrix grad --> dl/dw = dl/dy . xt
        print(f'lhs: {dense.size()}, rhs: {grad_output.size()}')
        grad_values = torch_sputnik.sddmm(m, k,
                                        row_indices, 
                                        row_offsets, 
                                        column_indices,
                                        dense.contiguous(),
                                        grad_output)

        values_t, row_offsets_t, column_indices_t = torch_sputnik.csr_transpose(m, k, 
                                                                                values, 
                                                                                row_offsets, 
                                                                                column_indices)
        
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad -->  dl/dx = wt . dl/dy
        grad_dense = torch_sputnik.spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output.contiguous())

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_dense

def copy_params(linear, sparse_linear):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(linear.weight.detach().clone())
    print(f'Row indices: {row_indices.size()}, Row offsets: {row_offsets.size()}')
    sparse_linear.values = nn.Parameter(values)
    sparse_linear.row_indices = row_indices
    sparse_linear.row_offsets = row_offsets
    sparse_linear.column_indices = column_indices
    
    #sparse_linear.bias = nn.Parameter(linear.bias.detach().clone())

if __name__ == '__main__':
    # w = (256, 128), x = (72, 128)
    m, k, n = 256, 128, 72
    input_features, output_features = k, m
    
    linear = nn.Linear(input_features, output_features, bias=False).cuda()
    #linear.bias = nn.Parameter(torch.ones_like(linear.bias))
    prune.random_unstructured(linear, name="weight", amount=0.9)
    #prune.remove(linear, 'weight')
    
    sparse_linear = SparseLinear(input_features, output_features).cuda()

    x = torch.randn(n, k).cuda()

    # [DENSE] --> x = (72, 128), w = (256, 128), op=x.wT, y = (72, 256)
    dense_output = linear(x)
    print(dense_output.size())

    copy_params(linear, sparse_linear)
    
    # [SPARSE] --> x = (72, 128), w = (256, 128), op=(w.xT)T y = (72, 256)
    sparse_output = sparse_linear(x)
    # print(sparse_output.size())

    # if (torch.abs(dense_output - sparse_output) < 1e-2).sum() == k * output_features:
    #     print("Output matches")
    # else:
    #     print("Doesn't match")

    # dummy = torch.ones_like(sparse_output)
    
    # dense_loss = (dummy - dense_output).sum()
    # dense_loss.backward()
    
    # sparse_loss = (dummy - sparse_output).sum()
    # sparse_loss.backward()
    
    # for name, param in linear.named_parameters():
    #     if name == 'weight':
    #         weight_grad = param.grad
    #     print(f'[DENSE] {name}:  {weight_grad}')
    
    # for name, param in sparse_linear.named_parameters():
    #     if name == 'values':
    #         values_grad = param.grad
    #     print(f'[SPARSE] {name}:  {values_grad}')
        
    # # if ((values_grad.reshape(m, n) - weight_grad) < 1e-2).sum() == m * n:
    # #     print("Grad matches")
    # # else:
    # #     print("Grad doesn't match")