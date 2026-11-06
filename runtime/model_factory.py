from runtime.registry import CompatibilityError, resolve_architecture


def resolve_adapter_class(config):
    """Reject known adapter stubs before opening checkpoint tensors."""
    from models import MODEL_REGISTRY
    spec = resolve_architecture(config)
    adapter = MODEL_REGISTRY[spec.adapter]
    missing = set(getattr(adapter, '__abstractmethods__', ()))
    if spec.streaming and not callable(getattr(adapter, 'create_layer_layout', None)):
        missing.add('create_layer_layout')
    if missing:
        names = ', '.join(sorted(missing))
        raise CompatibilityError(
            f'{config.architectures[0]} has an incomplete adapter (missing {names}); '
            'use a supported architecture or a compatible GGUF with llama_cpp.'
        )
    return adapter


def create_adapter(model, loader, config):
    return resolve_adapter_class(config)(model, loader)
