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
                'csrc/adjacency_builder.cu',
                'csrc/value_mutation_kernel.cu',
                'csrc/reorganize_kernel.cu',
                'csrc/crossover_kernel.cu', # <--- [수정] 이 줄을 추가해야 합니다.
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)