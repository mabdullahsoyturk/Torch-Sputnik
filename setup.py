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
            'src/spmm_cuda_graph.cu',
            'src/sddmm_cuda.cu',
            'src/sddmm_cuda_graph.cu',
            'src/softmax_cuda.cu',
            'src/transpose_cuda.cu'
        ],
        #library_dirs=['/home/soyturk/bind-sputnik'],
        libraries=['sputnik', 'cusparse'],
        extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'],
        extra_compile_args=extra_compile_args),
    ],
    include_dirs=["./include", "/usr/local/sputnik/include"],
    cmdclass={
        'build_ext': BuildExtension
    })
