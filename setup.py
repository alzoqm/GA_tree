# setup.py
from setuptools import setup
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
            # 디버깅이 필요할 경우 아래 주석을 해제하세요.
            # extra_compile_args={
            #     'cxx': ['-g'],
            #     'nvcc': ['-G', '-g']
            # }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)