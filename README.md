# Steps

* &#9989; Write Python bindnigs for SpMM with Pytorch tensors.
* Write Python bindnigs for SDDMM with Pytorch tensors. (this is needed for backward)
* Wrap SpMM and SDDMM with **torch.autograd.Function** and **torch.nn.Module** to make them first class citizens of PyTorch.

```Python
import sputnik

class MyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        outputs = sputnik.forward(input, weights, bias)
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx):
        outputs = sputnik.backward(*ctx.saved_tensors)
        return outputs


class MyLinear(torch.nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()

    def forward(self, input, state):
        return MyLinearFunction.apply(input, self.weights, self.bias)
```

* Replace **nn.Linear** and attention products with **sputnik.spmm**.