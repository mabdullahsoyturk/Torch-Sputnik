import torch
import torch_sputnik
from utils.util import *

class Spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnzs, row_indices, values, row_offsets, column_indices, dense, mask):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnzs = nnzs
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.mask = mask
        ctx.save_for_backward(values, dense)

        result = torch_sputnik.spmm(m, k, n, nnzs, row_indices, values, row_offsets, column_indices, dense)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnzs = ctx.nnzs
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        mask = ctx.mask
        values, dense = ctx.saved_tensors

        grad_m = grad_k = grad_n = grad_nnz = grad_row_indices = grad_values = grad_row_offsets = grad_column_indices = grad_mask = grad_dense = None

        # sparse matrix grad 
        grad_values = torch_sputnik.sddmm(m, k, n, nnzs, row_indices, row_offsets, column_indices, grad_output, dense, mask)

        values_t = values.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnzs, values, row_offsets, column_indices, values_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)

        # dense matrix grad
        grad_dense = torch_sputnik.spmm(k, m, n, nnzs, row_indices_t, values_t, row_offsets_t, column_indices_t, grad_output)

        return grad_m, grad_k, grad_n, grad_nnz, grad_row_indices, grad_values, grad_row_offsets, grad_column_indices, grad_mask, grad_dense

class Sddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k, n, nnzs, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, mask):
        ctx.m = m
        ctx.k = k
        ctx.n = n
        ctx.nnzs = nnzs
        ctx.row_indices = row_indices
        ctx.row_offsets = row_offsets
        ctx.column_indices = column_indices
        ctx.save_for_backward(lhs_matrix, rhs_matrix)
        return torch_sputnik.sddmm(m, k, n, nnzs, row_indices, row_offsets, column_indices, lhs_matrix, rhs_matrix, mask)

    @staticmethod
    def backward(ctx, grad_output):
        m = ctx.m
        k = ctx.k
        n = ctx.n
        nnzs = ctx.nnzs
        row_indices = ctx.row_indices
        row_offsets = ctx.row_offsets
        column_indices = ctx.column_indices
        lhs_matrix, rhs_matrix = ctx.saved_tensors

        grad_m = grad_k = grad_n = grad_nnzs = grad_row_indices = grad_row_offsets = grad_column_indices = grad_lhs = grad_rhs = grad_mask = None
        
        # lhs grad
        grad_lhs = torch_sputnik.spmm(m, k, n, nnzs, row_indices, grad_output, row_offsets, column_indices, rhs_matrix)

        grad_t = grad_output.clone()
        row_offsets_t = row_offsets.clone()
        column_indices_t = column_indices.clone()

        torch_sputnik.csr_transpose(m, n, nnzs, grad_output, row_offsets, column_indices, grad_t, row_offsets_t, column_indices_t)
        row_indices_t = diffsort(row_offsets_t)
        
        # rhs grad
        grad_rhs = torch_sputnik.spmm(n, k, m, nnzs, row_indices_t, grad_t, row_offsets_t, column_indices_t, lhs_matrix)

        return grad_m, grad_k, grad_n, grad_nnzs, grad_row_indices, grad_row_offsets, grad_column_indices, grad_lhs, grad_rhs, grad_mask

class SparseAttention(torch.nn.Module):
    def __init__(self, m, k, n, nnzs, row_indices, values, row_offsets, column_indices, q3d, k3d, v3d, mask):
        super().__init__()
        self.values = torch.nn.Parameter(values)
        self.q3d = torch.nn.Parameter(q3d)
        self.k3d = torch.nn.Parameter(k3d)
        self.v3d = torch.nn.Parameter(v3d)

        self.m = m
        self.k = k
        self.n = n
        self.nnzs = nnzs
        self.row_indices = row_indices
        self.row_offsets = row_offsets
        self.column_indices = column_indices
        self.mask = mask

        self.sddmm = Sddmm.apply
        self.spmm = Spmm.apply

    def forward(self):
        logits = self.sddmm(
                    self.m, self.k, self.n, self.nnzs,
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices, 
                    self.q3d, 
                    self.k3d,
                    self.mask
                )

        weights = torch_sputnik.softmax(
                    self.m, self.n, self.nnzs,
                    logits, 
                    self.row_indices, 
                    self.row_offsets, 
                    self.column_indices
                )

        out = self.spmm(
                self.m, self.k, self.n, self.nnzs,
                self.row_indices, 
                weights, 
                self.row_offsets, 
                self.column_indices, 
                self.v3d,
                self.mask
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
    values, row_indices, row_offsets, column_indices, nonzeros = dense_to_sparse(a)
    
    b = torch.arange(nnz + 1, (2 * nnz) + 1, dtype=torch.float32).view(m, k)
    values2, row_indices2, row_offsets2, column_indices2, nonzeros2 = dense_to_sparse(b)
    
    sparse_values = torch.cat((values, values2))
    sparse_row_indices = torch.cat((row_indices, row_indices2))
    sparse_row_offsets = torch.cat((row_offsets, row_offsets2))
    sparse_column_indices = torch.cat((column_indices, column_indices2))
    sparse_nonzeros = torch.cat((nonzeros, nonzeros2))

    q3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()
    k3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()
    v3d = torch.arange(1, 2 * (nnz + 1) - 1, dtype=torch.float32).view(replication, k, n).cuda()

    mask = torch.ones_like(sparse_values)

    model = SparseAttention(m, k, n, sparse_nonzeros, sparse_row_indices, sparse_values, sparse_row_offsets, sparse_column_indices, q3d, k3d, v3d, mask)

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