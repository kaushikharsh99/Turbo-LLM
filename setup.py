from setuptools import setup, find_packages

setup(
    name="turbo-llm",
    version="0.1.2",
    description="Turbo-LLM: Fast memory-efficient inference engine for large MoE models",
    author="kaushikharsh99",
    url="https://github.com/kaushikharsh99/Turbo-LLM",
    packages=find_packages(),
    package_data={"config": ["default.yaml"]},
    py_modules=["run", "phase2_generate"],
    install_requires=[
        "torch",
        "transformers==5.14.1",
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
    extras_require={"test": ["pytest>=8,<10"], "gguf": ["llama-cpp-python>=0.3.16,<0.4"]},
    python_requires=">=3.10",
)
