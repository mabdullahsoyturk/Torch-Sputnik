import torch
import torch_sputnik

a = torch.tensor([
    [1,2], 
    [0,0],
    [0,0]
])

b = torch.Tensor([
    [1,2,3],
    [1,2,3]
]).cuda()

c = torch.zeros((3,3)).cuda()

torch_sputnik.torch_spmm(a,b,c)

