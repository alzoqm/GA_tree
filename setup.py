# setup.py
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gatree_cuda',
    ext_modules=[
        CUDAExtension(
            name='gatree_cuda',
            sources=[
                'csrc/predict.cpp',
                'csrc/predict_kernel.cu',
            ],
            # extra_compile_args={
            #     'cxx': ['-g'], # 디버깅용
            #     'nvcc': ['-G', '-g'] # 디버깅용
            # }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)