from setuptools import setup, find_packages

setup(
    name="turbo-llm",
    version="0.1.0",
    description="Turbo-LLM: Fast memory-efficient inference engine for large MoE models",
    author="kaushikharsh99",
    url="https://github.com/kaushikharsh99/Turbo-LLM",
    packages=find_packages(),
    py_modules=["run", "phase2_generate"],
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
