# TorchSputnik

This repository contains PyTorch bindings for [Sputnik](https://github.com/google-research/sputnik) library. Sputnik is a sparse linear algebra library. It is a standalone C++ library. You can find the Tensorflow bindings here: [Tensorflow Bindings](https://github.com/google-research/google-research/tree/master/sgk/sparse/ops).

## Build

```Bash
docker build -t torchsputnik:latest .
docker run -it --runtime=nvidia torchsputnik:latest
```

## Run

```Bash
python3 tests/test_linear.py
```

## Main Operations
![SpMM and SDDMM](figures/spmm_and_sddmm.png)

### Sparse Matrix Matrix Multiplication (SpMM)

AxB = C where A is sparse. B and C are dense.

### Sampled Dense Dense Matrix Multiplication (SDDMM)

(AxB).C = D where A and B are dense. C and D are sparse.

## Using Sparse Linear Layer As A PyTorch Module

Check [modules/sparse_linear.py](modules/sparse_linear.py)

## Sputnik vs cuSPARSE Performance Comparison for SpMM

### A100 M=N=K=64

| Density | Sputnik   | CuSparse    | CuBlas    |
| ------- | --------- | ----------- | --------- |
| 0.5     | 0.007468  | 0.088439466 | 0.007091  |
| 0.25    | 0.005868  | 0.086323199 | 0.007091  |
| 0.2     | 0.005615  | 0.085640532 | 0.007091  |
| 0.15    | 0.005206  | 0.085503999 | 0.007091  |
| 0.1     | 0.004922  | 0.086323199 | 0.007091  |
| 0.05    | 0.004506  | 0.146739199 | 0.007091  |

### A100 M=N=K=4096

| Density | Sputnik   | CuSparse    | CuBlas    |
| ------- | --------- | ----------- | --------- |
| 0.5     | 13.047884 | 107.3758555 | 7.2852821 |
| 0.25    | 7.870807  | 53.62438799 | 7.2852821 |
| 0.2     | 6.065586  | 42.65809911 | 7.2852821 |
| 0.15    | 5.078734  | 32.06635513 | 7.2852821 |
| 0.1     | 4.023627  | 21.68115202 | 7.2852821 |
| 0.05    | 2.521871  | 10.97004381 | 7.2852821 |
