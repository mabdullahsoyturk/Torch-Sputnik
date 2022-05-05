import torch
import torch_sputnik
import numpy as np

def dense_to_sparse(matrix):
     """Converts dense numpy matrix to a csr sparse matrix."""
     csr = matrix.to_sparse_csr()
     values = csr.values().clone()
     row_offsets = csr.crow_indices().detach().to(torch.int32)
     row_indices = torch.from_numpy(np.argsort(-1 * np.diff(row_offsets.detach().numpy()))).to(torch.int32)
     column_indices = csr.col_indices().detach().to(torch.int32)

     return values.cuda(), row_indices.cuda(), row_offsets.cuda(), column_indices.cuda()

def diffsort(offsets):
  diffs = (offsets - torch.roll(offsets, -1, 0))[:-1]
  return torch.argsort(diffs, descending=True).to(torch.int32)

class Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnz, row_indices, values, row_offsets, column_indices, b):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnz = nnz
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(values, b)
        return torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, b)

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnz = ctx.nnz
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        values, b = ctx.saved_tensors

        grad_m = grad_k = grad_n = grad_nnz = grad_row_indices = grad_values = grad_row_offsets = grad_column_indices = grad_b = None
        grad_values = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, grad_output, b, values)

        values_t = values.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnz, values, row_offsets, column_indices, values_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)

        grad_b = torch_sputnik.spmm(k, m, n, nnz, row_indices_t, values_t, row_offsets_t, column_indices_t, grad_output)

        return grad_m, grad_k, grad_n, grad_nnz, grad_row_indices, grad_values, grad_row_offsets, grad_column_indices, grad_b

class Sddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnz, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, values):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnz = nnz
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)
        return torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, values)

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnz = ctx.nnz
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_m = grad_k = grad_n = grad_nnz = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = grad_values = None
        

        bias = torch.Tensor([])
        # lhs grad
        grad_lhs = torch_sputnik.spmm(m, k, n, nnz, row_indices, grad_output, row_offsets, column_indices, rhs_matrix, bias)

        grad_t = grad_output.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnz, grad_output, row_offsets, column_indices, grad_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)
        
        # rhs grad
        grad_rhs = torch_sputnik.spmm(n, k, m, nnz, row_indices_t, grad_t, row_offsets_t, column_indices_t, lhs_matrix, bias)
        #print("dense matrix grad:")
        #print(out)

        return grad_m, grad_k, grad_n, grad_nnz, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs, grad_values

class SparseAttention(torch.nn.Module):
    def __init__(self, m, k, n, nnz, row_indices, values, row_offsets, column_indices, q3d, k3d, v3d):
        super().__init__()
        self.values = values
        self.q3d = torch.nn.Parameter(q3d)
        self.k3d = torch.nn.Parameter(k3d)
        self.v3d = torch.nn.Parameter(v3d)

        self.m = m
        self.k = k
        self.n = n
        self.nnz = nnz
        self.row_indices = row_indices
        self.row_offsets = row_offsets
        self.column_indices = column_indices

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def forward(self):
        logits = self.sddmm(
            self.m, self.k, self.n, self.nnz, 
            self.row_indices, 
            self.row_offsets, 
            self.column_indices, 
            self.q3d, 
            self.k3d, 
            self.values
        )

        weights = torch_sputnik.softmax(
            self.m, self.n, self.nnz, 
            logits, 
            self.row_indices, 
            self.row_offsets, 
            self.column_indices)

        out = self.spmm(
            self.m, self.k, self.n, self.nnz, 
            self.row_indices, 
            weights, 
            self.row_offsets, 
            self.column_indices, 
            self.v3d
        )

        return out

if __name__ == "__main__":
    m, k, n, nnz = 8, 8, 8, 64

    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)
    b = torch.arange(nnz + 1, (2 * nnz) + 1, dtype=torch.float32).view(m, k)
    values2, row_indices2, row_offsets2, column_indices2 = dense_to_sparse(b)
    sparse_values = torch.cat((values, values2))
    sparse_row_indices = torch.cat((row_indices, row_indices2))
    sparse_row_offsets = torch.cat((row_offsets, row_offsets2))
    sparse_column_indices = torch.cat((column_indices, column_indices2))

    #q3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    #k3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    #v3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    
    q3d = torch.arange(1, nnz + 1, dtype=torch.float32).view(k, n).cuda()
    k3d = torch.arange(1, nnz + 1, dtype=torch.float32).view(k, n).cuda()
    v3d = torch.arange(1, nnz + 1, dtype=torch.float32).view(k, n).cuda()

    #model = SparseAttention(m, k, n, nnz, sparse_row_indices, sparse_values, sparse_row_offsets, sparse_column_indices, q3d, k3d, v3d)
    model = SparseAttention(m, k, n, nnz, row_indices, values, row_offsets, column_indices, q3d, k3d, v3d)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # supposedly the correct results
    y = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()

    for t in range(1):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model()

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(f'Loss: {loss.item()}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()