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
            'sputnik.cpp',
            'spmm_cuda.cu',
            'sddmm_cuda.cu'
        ],
        include_dirs=['/home/msoyturk/sputnik'],
        library_dirs=['/home/msoyturk/bind-sputnik'],
        libraries=['sputnik'],
        extra_compile_args=extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
