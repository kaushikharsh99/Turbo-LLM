import json
from pathlib import Path

config_path = Path("/home/harsh/.turbollm/models/Qwen_Qwen3.6-35B-A3B-FP8/config.json")

with open(config_path, "r") as f:
    cfg = json.load(f)

print("=" * 80)
print("CONFIG")
print("=" * 80)

for k, v in sorted(cfg.items()):
    print(f"{k:35}: {v}")