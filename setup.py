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
                'csrc/crossover_kernel.cu',
                'csrc/node_mutation_kernel.cu',   # <--- NEW
                'csrc/mutation_utils_kernel.cu',  # <--- NEW
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)