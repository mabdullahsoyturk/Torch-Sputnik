from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {'cxx' : ['-O2']}
extra_compile_args['nvcc'] = ['-O3',
                              '-gencode', 'arch=compute_80,code=compute_80',
                              '-gencode', 'arch=compute_80,code=sm_80'
                              ]

setup(
    name='torch_sputnik',
    ext_modules=[
        CUDAExtension('torch_sputnik', [
            'src/sputnik.cpp',
            'src/spmm_cuda.cu',
            'src/spmm_bias_relu_cuda.cu',
            'src/spmm_graph_cuda.cu',
            'src/sddmm_cuda.cu',
            'src/sddmm_graph_cuda.cu',
            'src/softmax_cuda.cu',
            'src/transpose_cuda.cu'
        ],
        libraries=['sputnik', 'cusparse'],
        extra_link_args=['-L/usr/local/lib'],
        extra_compile_args=extra_compile_args),
    ],
    include_dirs=["./include", "/usr/local/include"],
    cmdclass={
        'build_ext': BuildExtension
    })
