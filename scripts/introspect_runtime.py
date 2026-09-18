#!/usr/bin/env python3
"""Introspect the REAL signatures of the exact torch/transformers versions in
the built image, so the engine calls match the shipped API rather than an
assumed one. Run inside the container:
    podman run --rm -v $PWD:/app:ro localhost/decisionmaker-serve:local python scripts/introspect_runtime.py
"""
import inspect

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


def sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError) as e:
        return f"<no signature: {e}>"


def about(name, obj):
    print(f"## {name}")
    print(f"  module   : {getattr(obj, '__module__', '?')}")
    print(f"  signature: {sig(obj)}")
    print()


about("AutoTokenizer.from_pretrained", AutoTokenizer.from_pretrained)
about("AutoModel.from_pretrained", AutoModel.from_pretrained)
about("PreTrainedTokenizerBase.encode", PreTrainedTokenizerBase.encode)
about("PreTrainedTokenizerBase.__call__", PreTrainedTokenizerBase.__call__)

# settable pad_token? pad_token_id? eos_token_id?
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="main")
print("## instantiated Qwen3-0.6B-Instruct tokenizer")
print("  eos_token_id =", t.eos_token_id, " pad_token_id =", t.pad_token_id)
try:
    t.pad_token = t.eos_token
    print("  pad_token settable : True  -> pad_token_id now", t.pad_token_id)
except Exception as e:
    print("  pad_token settable : False ->", type(e).__name__, e)
print("  encode('abc', add_special_tokens=False) ->", t.encode("abc", add_special_tokens=False))
print("  encode('abc', return_tensors='pt') ->", t.encode("abc", add_special_tokens=False, return_tensors="pt"))

# What does AutoModel for Qwen3 give us?
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", revision="main")
print("## Qwen3 config")
print("  hidden_size =", getattr(cfg, 'hidden_size', None), " model_type =", getattr(cfg, 'model_type', None))

# BaseModelOutput last_hidden_state shape (torch available, no weights needed via from_config)
m = AutoModel.from_config(cfg)
print("## AutoModel.from_config(hidden_size=%s).__class__ = %s" % (cfg.hidden_size, type(m).__name__))
# NOTE: AutoModel.from_pretrained is (*model_args, **kwargs) — torch_dtype,
# attn_implementation, revision are forwarded via **kwargs to the real
# PreTrainedModel.from_pretrained, so they are NOT named params here. The
# service's load call is the authoritative test; this check only confirms the
# forward contract below.
print("  (from_pretrained accepts torch_dtype/attn_implementation/revision via **kwargs)")

# forward input args
fwd = getattr(m, "forward")
print("## model.forward accepts input_ids/attention_mask?")
print("  params:", list(inspect.signature(fwd).parameters)[:8])