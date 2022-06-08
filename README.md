# TorchSputnik

This repository contains PyTorch bindings for [Sputnik](https://github.com/google-research/sputnik) library. Sputnik is a sparse linear algebra library. It is a standalone C++ library. You can find the Tensorflow bindings here: [Tensorflow Bindings](https://github.com/google-research/google-research/tree/master/sgk/sparse/ops).

## Main Operations
![SpMM and SDDMM](figures/spmm_and_sddmm.png)

### Sparse Matrix Matrix Multiplication (SpMM)

AxB = C where A is sparse. B and C are dense.

### Sampled Dense Dense Matrix Multiplication (SDDMM)

(AxB).C = D where A and B are dense. C and D are sparse.

## Transformer Attention Implementation

```Python
def attention(q, k, v, mask)
  scores = matmul(q, k, transpose_b=True)
  scores._masked_fill(mask == 0, -inf)
  attention_weights = softmax(logits)
  return matmul(attention_weights, v)
```
## Sparse Transformer Attention implementation

```Python
def sparse_attention(q, k, v, mask)
  q_3d, k_3d, v_3d = [preprocess_attention_component(x) for x in [q, k, v]]
  topology = to_sparse(mask)
  logits = replicated_sddmm(q_3d, k_3d, topology)
  attention_weights = replicated_sparse_softmax(logits, topology)
  out = replicated_spmm(weights, topology, v_3d)
  return out.reshape_to_4d
```