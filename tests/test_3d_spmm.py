import torch
import torch_sputnik
from utils.util import *

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

        result = torch_sputnik.spmm(m, k, n, nnz, row_indices, values, row_offsets, column_indices, b)
        
        return result

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

        # sparse matrix grad 
        grad_values = torch_sputnik.sddmm(m, k, n, nnz, row_indices, row_offsets, column_indices, grad_output, b)

        values_t = values.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnz, values, row_offsets, column_indices, values_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_b = torch_sputnik.spmm(k, m, n, nnz, row_indices_t, values_t, row_offsets_t, column_indices_t, grad_output)

        return grad_m, grad_k, grad_n, grad_nnz, grad_row_indices, grad_values, grad_row_offsets, grad_column_indices, grad_b

class Sddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)
        return torch_sputnik.sddmm(m, k, n, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix)

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        n = ctx.n
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_m = grad_k = grad_n = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = None
        
        # lhs grad
        grad_lhs = torch_sputnik.spmm(m, k, n, row_indices, grad_output, row_offsets, column_indices, rhs_matrix)

        grad_t = grad_output.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, grad_output, row_offsets, column_indices, grad_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)
        
        # rhs grad
        grad_rhs = torch_sputnik.spmm(n, k, m, row_indices_t, grad_t, row_offsets_t, column_indices_t, lhs_matrix)

        return grad_m, grad_k, grad_n, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs

class SparseAttention(torch.nn.Module):
    def __init__(self, m, k, n, row_indices, values, row_offsets, column_indices, q3d, k3d, v3d):
        super().__init__()
        self.values = torch.nn.Parameter(values)
        self.q3d = torch.nn.Parameter(q3d)
        self.k3d = torch.nn.Parameter(k3d)
        self.v3d = torch.nn.Parameter(v3d)

        self.m = m
        self.k = k
        self.n = n
        self.row_indices = row_indices
        self.row_offsets = row_offsets
        self.column_indices = column_indices

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def forward(self):
        logits = self.sddmm(
                    self.m, self.k, self.n, 
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices, 
                    self.q3d, 
                    self.k3d
                )

        weights = torch_sputnik.softmax(
                    self.m, self.n, 
                    logits, 
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices
                )

        out = self.spmm(
                self.m, self.k, self.n, 
                self.row_indices, 
                weights, 
                self.row_offsets, 
                self.column_indices, 
                self.v3d
            )

        return out

class Attention(torch.nn.Module):
    def __init__(self, q3d, k3d, v3d):
        super().__init__()
        self.q3d = torch.nn.Parameter(q3d)
        self.k3d = torch.nn.Parameter(k3d)
        self.v3d = torch.nn.Parameter(v3d)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self):
        scores = torch.matmul(self.q3d, self.k3d.transpose(-2, -1))
        
        attention_weights = self.softmax(scores)

        intermediate_token_representations = torch.matmul(attention_weights, self.v3d)
        #print(intermediate_token_representations)

        return intermediate_token_representations

def train_sparse():
    m, k, n, nnz = 8, 8, 8, 64
    replication = 2

    a = torch.arange(1, nnz + 1, dtype=torch.float32).view(m, k)
    values, row_indices, row_offsets, column_indices = dense_to_sparse(a)
    
    b = torch.arange(nnz + 1, (2 * nnz) + 1, dtype=torch.float32).view(m, k)
    values2, row_indices2, row_offsets2, column_indices2 = dense_to_sparse(b)
    
    sparse_values = torch.stack((values, values2))
    sparse_row_indices = torch.stack((row_indices, row_indices2))
    sparse_row_offsets = torch.stack((row_offsets, row_offsets2))
    sparse_column_indices = torch.stack((column_indices, column_indices2))

    q3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()
    k3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()
    v3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()

    model = SparseAttention(m, k, n, sparse_row_indices, sparse_values, sparse_row_offsets, sparse_column_indices, q3d, k3d, v3d)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # supposedly the correct results
    y = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()

    for t in range(1):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model()
        #print(y_pred)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(f'Loss: {loss.item()}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_normal():
    n, k, nnz = 8, 8, 64
    q3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    k3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()
    v3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()

    model = Attention(q3d, k3d, v3d)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # supposedly the correct results
    y = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(2, k, n).cuda()

    for t in range(1):
        y_pred = model()

        loss = criterion(y_pred, y)
        print(f'Loss: {loss.item()}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train_sparse()
    train_normal()