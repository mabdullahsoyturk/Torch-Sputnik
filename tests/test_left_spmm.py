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

     return values, row_indices, row_offsets, column_indices

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, x):
        print(f'x: {x.size()}, values: {self.values.size()}, row_indices: {self.row_indices.size()}')
        return SparseLinearFunction.apply(3072, 768, self.values, self.row_indices, self.row_offsets, self.column_indices, self.bias, x.transpose(1, 2).contiguous())

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, values, row_indices, row_offsets, column_indices, bias, dense):
        ctx.m = m
        ctx.k = k
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, dense)

        result = torch_sputnik.left_spmm(m, k, values, row_indices, row_offsets, column_indices, dense).transpose(1,2) + bias

        #print(f'result size: {result.size()}')

        return result

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_values = grad_row_indices = grad_row_offsets = grad_column_indices = grad_bias = grad_dense = None

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
        grad_dense = torch_sputnik.left_spmm(k, m, 
                                        values_t, 
                                        row_indices_t, 
                                        row_offsets_t, 
                                        column_indices_t, 
                                        grad_output.contiguous())

        grad_bias = grad_output.sum(0)

        return grad_m, grad_k, grad_values, grad_row_indices, grad_row_offsets, grad_column_indices, grad_bias, grad_dense

def copy_params(linear, sparse_linear):
    values, row_indices, row_offsets, column_indices = dense_to_sparse(linear.weight.detach().clone())
    sparse_linear.values = nn.Parameter(values)
    sparse_linear.row_indices = row_indices
    sparse_linear.row_offsets = row_offsets
    sparse_linear.column_indices = column_indices
    
    sparse_linear.bias = nn.Parameter(linear.bias.detach().clone())

if __name__ == '__main__':
    batch_size = 4
    m, k = 512, 768
    input_features, output_features = 768, 3072
    
    linear = nn.Linear(input_features, output_features).cuda()
    linear.bias = nn.Parameter(torch.ones_like(linear.bias))
    print(f'Linear weight: {linear.weight.size()}, Linear bias: {linear.bias.size()}')
    #prune.random_unstructured(linear, name="weight", amount=0.9)
    #prune.remove(linear, 'weight')
    
    sparse_linear = SparseLinear(input_features, output_features).cuda()

    x = torch.randn(batch_size, m, k).cuda()

    dense_output = linear(x)
    print(dense_output)

    copy_params(linear, sparse_linear)
    
    sparse_output = sparse_linear(x)
    print(sparse_output)

    if ((abs(dense_output) - abs(sparse_output)) < 1e-2).sum() == batch_size * m * output_features:
        print("Output matches")
    else:
        print("Doesn't match")
    torch.cuda.synchronize()
    exit()

    dummy = torch.ones_like(sparse_output)
    
    dense_loss = (dummy - dense_output).sum()
    dense_loss.backward()
    
    sparse_loss = (dummy - sparse_output).sum()
    sparse_loss.backward()
    
    for name, param in linear.named_parameters():
        if name == 'weight':
            weight_grad = param.grad
            print(param)
        print(f'Param Name: {name}:  {param.grad}')
    
    for name, param in sparse_linear.named_parameters():
        if name == 'values':
            values_grad = param.grad
        print(f'Param Name: {name}:  {param.grad}')
        
    if ((abs(values_grad.reshape(m, n)) - abs(weight_grad)) < 1e-2).sum() == batch_size * m * output_features:
        print("Grad matches")
    else:
        print("Grad doesn't match")