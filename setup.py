from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {'cxx' : ['-O2']}
extra_compile_args['nvcc'] = ['-O3',
                              '-gencode', 'arch=compute_86,code=compute_86',
                              '-gencode', 'arch=compute_86,code=sm_86'
                              ]

setup(
    name='torch_sputnik',
    ext_modules=[
        CUDAExtension('torch_sputnik', [
            'src/sputnik.cpp',
            'src/spmm_cuda.cu',
            'src/replicated_spmm_cuda.cu'
            'src/sddmm_cuda.cu',
            'src/softmax_cuda.cu',
            'src/transpose_cuda.cu'
        ],
        include_dirs=['/home/msoyturk/sputnik', '/cm/shared/apps/cuda11.3/toolkit/11.3.0/include'],
        library_dirs=['/home/msoyturk/bind-sputnik', '/cm/shared/apps/cuda11.3/toolkit/11.3.0/lib64'],
        libraries=['sputnik', 'cusparse'],
        extra_compile_args=extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
