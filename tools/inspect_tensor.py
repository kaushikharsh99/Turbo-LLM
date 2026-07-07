from pathlib import Path

from safetensors import safe_open

MODEL_DIR = Path("/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8")

files = sorted(MODEL_DIR.glob("*.safetensors"))

print("=" * 80)
print("SAFE TENSORS")
print("=" * 80)

for file in files:

    print()
    print(file.name)

    with safe_open(file, framework="pt") as f:

        for name in f.keys():

            tensor = f.get_tensor(name)

            print(
                f"{name:60}"
                f"{str(tuple(tensor.shape)):20}"
                f"{tensor.dtype}"
            )