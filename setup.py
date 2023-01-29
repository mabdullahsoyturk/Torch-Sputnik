import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_directory = os.getcwd()[:-14]
sputnik_directory = current_directory + "/sputnik/build"

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
            'src/left_replicated_spmm.cu',
            'src/sddmm_cuda.cu',
            'src/softmax_cuda.cu',
            'src/transpose_cuda.cu',
        ],
        libraries=['sputnik', 'cusparse'],
        extra_link_args=['-L/usr/local/lib', f'-L{sputnik_directory}/lib'],
        extra_compile_args=extra_compile_args),
    ],
    include_dirs=["./include", "/usr/local/include", f'{sputnik_directory}/include'],
    cmdclass={
        'build_ext': BuildExtension
    })
