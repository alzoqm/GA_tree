# setup.py (수정됨)
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
                'csrc/adjacency_builder.cu', # [신규] 빌더 소스 파일 추가
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)