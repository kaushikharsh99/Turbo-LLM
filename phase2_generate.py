import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from loader.expert_loader import ExpertLoader
from runtime.model_factory import create_adapter
from runtime.engine import TurboEngine

@torch.no_grad()
def generate(
        model,
        prompt,
        max_new_tokens=50,
        temperature=0.0,
        top_p=1.0,
        config=None,
        chat=False,
        system_prompt=None,
        collector=None,
        thinking="off",
    ):
    if max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require max_new_tokens >= 0, temperature >= 0 and 0 < top_p <= 1")
    if max_new_tokens == 0:
        return ""
    from runtime.registry import CompatibilityError
    backend = (config or {}).get("execution", {}).get("backend", "auto")
    is_gguf = str(model).lower().endswith(".gguf")
    if backend not in {"auto", "turbo", "llama_cpp"}:
        raise CompatibilityError(f"Unknown backend: {backend}")
    if backend == "turbo" and is_gguf:
        raise CompatibilityError("Turbo streaming does not implement native GGUF kernels; select llama_cpp")
    if backend == "llama_cpp" and not is_gguf:
        raise CompatibilityError("llama_cpp requires a local .gguf file")
    if is_gguf:
        from runtime.gguf import generate_gguf
        return generate_gguf(model, prompt, max_new_tokens, temperature, top_p,
                             config, chat, system_prompt, collector, thinking)
    from runtime.registry import resolve_architecture
    from runtime.policy import ExecutionPolicy
    from types import SimpleNamespace
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=False)
    spec = resolve_architecture(hf_config)
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
        if collector is not None:
            collector.num_layers = adapter.num_layers
        engine = TurboEngine(adapter)
        return engine.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, config=config, chat=chat, system_prompt=system_prompt,
            collector=collector, thinking=thinking)
    finally:
        if loader is not None:
            loader.close()



@torch.no_grad()
def main():
    print("=" * 60)
    print("PHASE 2.2 — AUTOREGRESSIVE GENERATION WITH KV CACHE")
    print("=" * 60)

    prompt = "Choose any topic and write an informative article of approximately 500 words. Structure it with an introduction, explanation, examples, analysis, and conclusion. Make it engaging, factual, and coherent."
    generate("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", prompt, max_new_tokens=500)


if __name__ == "__main__":
    main()
