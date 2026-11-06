"""Optional llama.cpp execution. This is not Turbo expert streaming.

llama.cpp owns GGUF metadata, tokenizer, tensor mapping and quantized kernels.
"""
from pathlib import Path
from runtime.registry import CompatibilityError


def generate_gguf(model, prompt, max_new_tokens=50, temperature=0.0, top_p=1.0,
                  config=None, chat=False, system_prompt=None, collector=None,
                  thinking="off"):
    if collector is not None:
        raise CompatibilityError("GGUF backend does not expose Turbo routing collection")
    if thinking != "off":
        raise CompatibilityError("GGUF thinking control is not supported; use the model's chat template")
    path = Path(model)
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise CompatibilityError(f"Not a GGUF checkpoint: {path}")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise CompatibilityError("GGUF requires pip install 'turbo-llm[gguf]'") from exc
    options = (config or {}).get("gguf", {})
    llm = Llama(model_path=str(path), n_ctx=options.get("n_ctx", 512),
                n_gpu_layers=options.get("n_gpu_layers", 0),
                seed=options.get("seed", 0), n_batch=options.get("n_batch", 128),
                n_threads=options.get("n_threads", 4),
                use_mmap=True, use_mlock=False,
                verbose=options.get("verbose", False))
    try:
        kwargs = dict(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        if chat:
            if "tokenizer.chat_template" not in llm.metadata:
                raise CompatibilityError("GGUF has no chat template; use plain completion")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            result = llm.create_chat_completion(messages=messages, **kwargs)
            return result["choices"][0]["message"]["content"]
        return llm.create_completion(prompt=prompt, **kwargs)["choices"][0]["text"]
    finally:
        llm.close()
