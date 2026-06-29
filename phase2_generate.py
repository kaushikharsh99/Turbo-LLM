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
        config=None,
        chat=False,
        system_prompt=None,
        collector=None,
    ):
    # 1. Initialize Loader
    print(f"Loading weights index from {model}...")
    loader = ExpertLoader(model, config=config)
    
    # 2. Load Config and Meta Model
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    
    # Determine dtype from config
    dtype = torch.float16
    if config and "execution" in config and "dtype" in config["execution"]:
        dtype_str = config["execution"]["dtype"]
        if dtype_str in ("bf16", "bfloat16"):
            dtype = torch.bfloat16
        elif dtype_str in ("fp32", "float32"):
            dtype = torch.float32

    # Check if MoE from config architecture
    arch = hf_config.architectures[0]
    is_moe = "Moe" in arch or "moe" in arch.lower()

    if is_moe:
        with init_empty_weights():
            causal_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, torch_dtype=dtype)
    else:
        causal_model = AutoModelForCausalLM.from_pretrained(
            model,
            config=hf_config,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(loader.DEVICE)

    # 3. Instantiate adapter
    adapter = create_adapter(causal_model, loader, hf_config)

    if collector is not None:
        collector.num_layers = adapter.num_layers

    # 4. Instantiate TurboEngine and Generate
    engine = TurboEngine(adapter)
    return engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        config=config,
        chat=chat,
        system_prompt=system_prompt,
        collector=collector,
    )


@torch.no_grad()
def main():
    print("=" * 60)
    print("PHASE 2.2 — AUTOREGRESSIVE GENERATION WITH KV CACHE")
    print("=" * 60)

    prompt = "Choose any topic and write an informative article of approximately 500 words. Structure it with an introduction, explanation, examples, analysis, and conclusion. Make it engaging, factual, and coherent."
    generate("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", prompt, max_new_tokens=500)


if __name__ == "__main__":
    main()
