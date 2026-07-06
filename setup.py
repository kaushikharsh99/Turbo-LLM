from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="turbo-llm",
    version="0.1.2",
    description="Turbo-LLM: Fast memory-efficient inference engine for large MoE models",
    author="kaushikharsh99",
    url="https://github.com/kaushikharsh99/Turbo-LLM",
    packages=find_packages(),
    py_modules=["run", "phase2_generate"],
    ext_modules=[
        CppExtension(
            name="turbollm_cpp",
            sources=[
                "backend/bindings.cpp",
                "backend/moe_executor.cpp",
                "backend/grouped_gemm.cpp",
                "backend/dequant_cache.cpp",
                "backend/fp8_dequant.cpp",
                "backend/tensor_utils.cpp",
            ],
            extra_compile_args=["-std=c++20"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "psutil",
        "pyyaml",
        "huggingface_hub",
    ],
    entry_points={
        "console_scripts": [
            "turbo-llm=run:main",
        ],
    },
    python_requires=">=3.8",
)
