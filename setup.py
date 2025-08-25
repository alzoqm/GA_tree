# setup.py (수정됨)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gatree_cuda',
    ext_modules=[
        # Core prediction module
        CUDAExtension(
            name='gatree_predict',
            sources=[
                'csrc/predict_bindings.cpp',
                'csrc/predict_kernel.cu',
                'csrc/adjacency_builder.cu',
            ],
        ),
        # Mutation operations
        CUDAExtension(
            name='gatree_mutation',
            sources=[
                'csrc/mutation_bindings.cpp',
                'csrc/value_mutation_kernel.cu',
                'csrc/node_mutation_kernel.cu',
                'csrc/subtree_mutation_kernel.cu',
                'csrc/mutation_utils_kernel.cu',
            ],
        ),
        # Crossover operations
        CUDAExtension(
            name='gatree_crossover',
            sources=[
                'csrc/crossover_bindings.cpp',
                'csrc/crossover_kernel.cu',
                'csrc/node_crossover_kernel.cu',
                'csrc/subtree_crossover_kernel.cu',
                'csrc/root_crossover_kernel.cu',
            ],
        ),
        # Utility operations
        CUDAExtension(
            name='gatree_utils',
            sources=[
                'csrc/utils_bindings.cpp',
                'csrc/reorganize_kernel.cu',
                'csrc/validate_kernel.cu',
                'csrc/make_population.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    verbose=True
)
