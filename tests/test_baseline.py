from types import SimpleNamespace
import pytest
import torch
from runtime.registry import resolve_architecture, CompatibilityError
from runtime.policy import ExecutionPolicy
from loader.checkpoint import SafetensorsReader
from cache.kv_cache import KVCache


def test_registry_exact():
    assert resolve_architecture(SimpleNamespace(architectures=['LlamaForCausalLM'])).streaming is False
    assert resolve_architecture(SimpleNamespace(architectures=['Qwen3MoeForCausalLM'])).streaming
    with pytest.raises(CompatibilityError):
        resolve_architecture(SimpleNamespace(architectures=['FakeLlamaForCausalLM']))


def test_unsharded_reader(tmp_path):
    from safetensors.torch import save_file
    save_file({'weight': torch.eye(2)}, tmp_path / 'model.safetensors')
    reader = SafetensorsReader(tmp_path)
    torch.testing.assert_close(reader.get_tensor('weight'), torch.eye(2))


def test_cache_append():
    cache = KVCache(4)
    x = torch.randn(1, 2, 2, 4)
    cache.update(x, x, 0)
    key, _ = cache.update(x[:, :, :1], x[:, :, :1], 0)
    torch.testing.assert_close(key, torch.cat([x, x[:, :, :1]], dim=2))
    assert cache.get_seq_length() == 3


def test_policy():
    assert ExecutionPolicy.from_config({'execution': {'device': 'cpu', 'dtype': 'fp32'}}).dtype == torch.float32


def test_tiny_llama_generation(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from phase2_generate import generate
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32,
                         bos_token_id=1, eos_token_id=None, pad_token_id=0)
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    raw = Tokenizer(WordLevel({'[PAD]': 0, '[BOS]': 1, '[UNK]': 2, 'hello': 3,
                               'world': 4, 'a': 5, 'b': 6, 'c': 7}, unk_token='[UNK]'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.save_pretrained(tmp_path)
    ids = tokenizer('hello world', return_tensors='pt').input_ids
    with torch.no_grad():
        expected = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    text = generate(str(tmp_path), 'hello world', max_new_tokens=3,
                    config={'execution': {'device': 'cpu', 'dtype': 'fp32'}})
    assert text == tokenizer.decode(expected[0, ids.shape[1]:].tolist())


def test_qwen_individual_expert_routing(tmp_path):
    import json
    from safetensors.torch import save_file
    from loader.expert_loader import ExpertLoader
    from execution.router import RouterExecutor
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_moe'}))
    gate = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    weights = {'model.layers.0.mlp.gate.weight': gate}
    for expert in range(3):
        for projection in ['gate_proj', 'up_proj', 'down_proj']:
            weights[f'model.layers.0.mlp.experts.{expert}.{projection}.weight'] = torch.eye(2)
    save_file(weights, tmp_path / 'model.safetensors')
    loader = ExpertLoader(str(tmp_path), config={'execution': {'device': 'cpu', 'dtype': 'fp32'},
                                                'cache': {'expert_limit': 2},
                                                'memory': {'max_vram_mb': 100}})
    try:
        router = RouterExecutor(loader, 1, top_k=2)
        indices, scores = router.compute_routing(0, torch.tensor([[[2., 1.]]]))
        assert indices.tolist() == [[0, 1]]
        torch.testing.assert_close(scores, torch.softmax(torch.tensor([[2., 1.]]), -1))
        assert loader.cache_limit == 2
        assert loader.max_vram_gb == 100 / 1024
    finally:
        loader.close()


def test_gguf_backend_contract(tmp_path, monkeypatch):
    import sys
    from runtime.gguf import generate_gguf
    path = tmp_path / 'tiny.gguf'
    path.write_bytes(b'GGUF')
    closed = []
    class FakeLlama:
        def __init__(self, **kwargs):
            assert kwargs['model_path'] == str(path)
            assert kwargs['n_gpu_layers'] == 0
        def create_completion(self, **kwargs):
            assert kwargs['max_tokens'] == 2
            return {'choices': [{'text': 'result'}]}
        def close(self):
            closed.append(True)
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=FakeLlama))
    assert generate_gguf(path, 'hello', max_new_tokens=2) == 'result'
    assert closed == [True]


@pytest.mark.parametrize("packed", [False, True])
def test_qwen_streaming_logits(tmp_path, packed):
    import json
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from safetensors.torch import save_file
    from loader.expert_loader import ExpertLoader
    from runtime.model_factory import create_adapter
    from runtime.engine import TurboEngine
    torch.manual_seed(3)
    config = Qwen3MoeConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, num_experts=3, num_experts_per_tok=2,
        max_position_embeddings=32)
    reference = Qwen3MoeForCausalLM(config).eval()
    config.architectures = ['Qwen3MoeForCausalLM']
    config.to_json_file(tmp_path / 'config.json')
    weights = {}
    for name, tensor in reference.state_dict().items():
        if packed:
            weights[name] = tensor
        elif name.endswith('experts.gate_up_proj'):
            prefix = name.removesuffix('gate_up_proj')
            for expert, matrix in enumerate(tensor):
                gate, up = matrix.chunk(2, dim=0)
                weights[f'{prefix}{expert}.gate_proj.weight'] = gate.contiguous()
                weights[f'{prefix}{expert}.up_proj.weight'] = up.contiguous()
        elif name.endswith('experts.down_proj'):
            prefix = name.removesuffix('down_proj')
            for expert, matrix in enumerate(tensor):
                weights[f'{prefix}{expert}.down_proj.weight'] = matrix.contiguous()
        else:
            weights[name] = tensor
    save_file(weights, tmp_path / 'model.safetensors')
    loader = ExpertLoader(str(tmp_path), {'execution': {'device': 'cpu', 'dtype': 'fp32'},
                                        'cache': {'expert_limit': 1}})
    from accelerate import init_empty_weights
    with init_empty_weights():
        skeleton = Qwen3MoeForCausalLM(config)
    adapter = create_adapter(skeleton.eval(), loader, config)
    try:
        engine = TurboEngine(adapter)
        cache = adapter.create_cache(8)
        from transformers import DynamicCache
        reference_cache = DynamicCache(config=config)
        with torch.no_grad():
            for ids in [torch.tensor([[3, 4]]), torch.tensor([[5]])]:
                start = cache.get_seq_length()
                pos = torch.arange(start, start + ids.shape[1]).unsqueeze(0)
                hidden = adapter.embed(ids)
                rotary = adapter.rotary_embeddings(hidden, pos)
                mask = adapter.create_attention_mask(hidden, cache, pos)
                hidden, _ = engine.layer.execute_layer(0, hidden, mask, pos, rotary, cache)
                logits = adapter.lm_head(adapter.final_norm(hidden))
                expected = reference(ids, past_key_values=reference_cache, use_cache=True).logits
                torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-4)
    finally:
        loader.close()


def test_real_gguf_when_available():
    import os
    from pathlib import Path
    path = Path(os.environ.get('TURBO_TEST_GGUF', '.local-models/stories15M-q4_0.gguf'))
    if not path.is_file():
        pytest.skip('Set TURBO_TEST_GGUF to run actual quantized inference')
    pytest.importorskip('llama_cpp')
    from llama_cpp import Llama
    from phase2_generate import generate
    options = {'n_ctx': 128, 'n_gpu_layers': 0, 'n_threads': 4, 'n_batch': 128, 'seed': 0}
    from contextlib import closing
    with closing(Llama(model_path=str(path), verbose=False, **options)) as reference:
        expected = reference.create_completion(prompt='Once upon a time', max_tokens=8,
                                                temperature=0., top_p=1.)['choices'][0]['text']
    actual = generate(str(path), 'Once upon a time', max_new_tokens=8,
                      config={'gguf': options})
    assert actual == expected
    assert actual.strip()


def test_cache_policy_rejects_invalid_budgets():
    from runtime.policy import CachePolicy
    for config in [{'cache': {'expert_limit': 0}}, {'memory': {'max_ram_percent': -1}},
                   {'memory': {'max_vram_mb': 0}}]:
        with pytest.raises(ValueError):
            CachePolicy.from_config(config)


def test_loader_cleanup_on_generation_error(tmp_path, monkeypatch):
    # An exception during model construction must release checkpoint resources.
    import phase2_generate
    closed = []
    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr(phase2_generate.AutoConfig, 'from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(architectures=['Qwen3MoeForCausalLM']))
    monkeypatch.setattr(phase2_generate, 'ExpertLoader', FakeLoader)
    def fail(*args, **kwargs):
        raise RuntimeError('construction failed')
    monkeypatch.setattr(phase2_generate.AutoModelForCausalLM, 'from_config', fail)
    with pytest.raises(RuntimeError, match='construction failed'):
        phase2_generate.generate(str(tmp_path), 'hello')
    assert closed == [True]
