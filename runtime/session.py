"""Own native model/loader lifetime independently of generation requests."""
from contextlib import contextmanager
from types import SimpleNamespace
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from loader.expert_loader import ExpertLoader
from runtime.policy import ExecutionPolicy
from runtime.registry import resolve_architecture, CompatibilityError
from runtime.model_factory import create_adapter, resolve_adapter_class
from runtime.engine import TurboEngine


@contextmanager
def open_turbo_session(model, config=None):
    """Load once, generate repeatedly, then release storage/cache resources."""
    if str(model).lower().endswith('.gguf'):
        raise CompatibilityError('Native sessions require a safetensors directory; GGUF uses llama_cpp')
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=False)
    spec = resolve_architecture(hf_config)
    resolve_adapter_class(hf_config)
    policy = ExecutionPolicy.from_config(config)
    loader = None
    try:
        if spec.streaming:
            loader = ExpertLoader(model, config=config)
            with init_empty_weights():
                causal_model = AutoModelForCausalLM.from_config(
                    hf_config, trust_remote_code=False, dtype=policy.dtype)
        else:
            loader = SimpleNamespace(DEVICE=policy.device, snapshot_path=model, close=lambda: None)
            causal_model = AutoModelForCausalLM.from_pretrained(
                model, config=hf_config, trust_remote_code=False, dtype=policy.dtype
            ).to(policy.device)
        causal_model.eval()
        adapter = create_adapter(causal_model, loader, hf_config)
        engine = TurboEngine(adapter)
        try:
            yield engine
        finally:
            # A retained engine should not keep a closed model or collector alive.
            engine.adapter = None
            engine.layer = None
            engine.router = None
            engine.moe = None
    finally:
        if loader is not None:
            loader.close()
