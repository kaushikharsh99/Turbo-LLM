from pathlib import Path
import yaml


def load_config(path=None):
    if path is None:
        path = Path(__file__).with_name("default.yaml")
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("model", "runtime", "execution", "memory", "cache", "gguf"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
    return config
