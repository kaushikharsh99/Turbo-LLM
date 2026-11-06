
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
    from runtime.options import validate_generation
    validate_generation(max_new_tokens, temperature, top_p)
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
    from runtime.session import open_turbo_session
    with open_turbo_session(model, config) as engine:
        if collector is not None:
            collector.num_layers = engine.adapter.num_layers
        return engine.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, config=config, chat=chat, system_prompt=system_prompt,
            collector=collector, thinking=thinking)



def main():
    print("=" * 60)
    print("PHASE 2.2 — AUTOREGRESSIVE GENERATION WITH KV CACHE")
    print("=" * 60)

    prompt = "Choose any topic and write an informative article of approximately 500 words. Structure it with an introduction, explanation, examples, analysis, and conclusion. Make it engaging, factual, and coherent."
    generate("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", prompt, max_new_tokens=500)


if __name__ == "__main__":
    main()
