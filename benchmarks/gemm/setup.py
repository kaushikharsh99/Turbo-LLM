import os
from setuptools import setup
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda compiler_name, compiler_version: True
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cutlass_inc = os.path.join(repo_dir, "third_party", "cutlass", "include")
cutlass_tools_inc = os.path.join(repo_dir, "third_party", "cutlass", "tools", "util", "include")

setup(
    name="cutlass_gemm_benchmark",
    ext_modules=[
        CppExtension(
            name="cutlass_gemm_benchmark",
            sources=["cutlass_gemm.cpp"],
            include_dirs=[cutlass_inc, cutlass_tools_inc, "/usr/local/cuda/include"],
            library_dirs=["/usr/local/cuda/lib64"],
            libraries=["cudart", "cublas", "cublasLt"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
