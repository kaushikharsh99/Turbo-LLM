import json
from collections import Counter
from pathlib import Path

index_path = Path("/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8/model.safetensors.index.json")

with open(index_path, "r") as f:
    index = json.load(f)

weight_map = index["weight_map"]

print("=" * 80)
print("GENERAL")
print("=" * 80)

print(f"Total tensors : {len(weight_map)}")

print()

print("=" * 80)
print("SHARDS")
print("=" * 80)

counter = Counter(weight_map.values())

for shard, count in sorted(counter.items()):
    print(f"{shard:40} {count}")

print()

print("=" * 80)
print("FIRST 100 TENSORS")
print("=" * 80)

for i, (name, shard) in enumerate(weight_map.items()):

    print(f"{name:60} -> {shard}")

    if i == 2999:
        break