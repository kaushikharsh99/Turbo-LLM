from runtime.registry import resolve_architecture


def create_adapter(model, loader, config):
    from models import MODEL_REGISTRY
    spec = resolve_architecture(config)
    return MODEL_REGISTRY[spec.adapter](model, loader)
