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

* Replace original attention products with sparse implementation.

Original implementation:

```Python
  logits = matmul(q, k, transpose_b=True)
  logits = add(logits, bias)
  weights = softmax(logits)
  return matmul(weights, v)
```
Sparse implementation:

```Python
  q_3d, k_3d, v_3d = [preprocess_attention_component(x) for x in [q, k, v]]
  logits = replicated_sddmm(q_3d, k_3d, topology, transpose_rhs=True)
  weights = replicated_sparse_softmax(logits, topology)
  out = replicated_spmm(weights, topology, v_3d)
  return reshape(out, tf.shape(q))
```