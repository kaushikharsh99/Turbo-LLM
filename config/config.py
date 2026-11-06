from copy import deepcopy
from pathlib import Path
import yaml


def _read_mapping(path):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    for section in ('model', 'runtime', 'execution', 'memory', 'cache', 'gguf'):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f'Configuration section {section} must be a mapping')
    return config


def load_config(path=None):
    """Overlay user sections onto packaged defaults, independent of cwd."""
    config = deepcopy(_read_mapping(Path(__file__).with_name('default.yaml')))
    if path is not None:
        for section, value in _read_mapping(path).items():
            if isinstance(value, dict) and isinstance(config.get(section), dict):
                config[section].update(value)
            else:
                config[section] = value
    return config
