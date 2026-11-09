"""Frozen PRM800K segment scorer for PolySeg.

The Qwen PRM checkpoint is a token classifier: it predicts the probability
that the step ending at ``<extra_0>`` is correct.  This adapter evaluates the
prefix ending at each rollout-owned segment boundary and returns that
probability as a utility/potential.  PolySeg's ``utility_delta`` protocol then
turns adjacent potentials into macro rewards.

The PRM score is deliberately labelled as a model-derived pseudo-label.  It
is not substituted for the exact GSM8K verifier, which remains available in
``final_accuracy`` on the last segment.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from verl.trainer.ppo.polyseg import feedback_protocol_contract
from verl.utils.reward_score import default_compute_score


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, int], tuple[Any, Any, torch.device]] = {}


class _ModernPRM(nn.Module):
    """Qwen PRM head on the current transformers Qwen2 implementation.

    The checkpoint's remote model code uses a removed ``DynamicCache`` API in
    transformers 5.x and produces NaNs in its old rotary path.  The backbone
    weights are unchanged, so loading them into the maintained Qwen2Model and
    restoring the two-layer classification head is equivalent and finite.
    """

    def __init__(self, backbone: nn.Module, score: nn.Module):
        super().__init__()
        self.model = backbone
        self.score = score

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs).last_hidden_state
        return type("PRMOutput", (), {"logits": self.score(hidden)})()


def _unwrap(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _messages(raw_prompt: Any, prefix: str) -> list[dict[str, str]]:
    """Build the PRM conversation while preserving the original user prompt."""

    raw_prompt = _unwrap(raw_prompt)
    if isinstance(raw_prompt, (list, tuple)):
        messages = []
        for message in raw_prompt:
            message = _unwrap(message)
            if isinstance(message, dict) and "role" in message and "content" in message:
                messages.append({"role": str(message["role"]), "content": str(message["content"])})
    elif isinstance(raw_prompt, str):
        messages = [{"role": "user", "content": raw_prompt}]
    else:
        messages = [{"role": "user", "content": str(raw_prompt)}]
    if not messages or messages[-1].get("role") != "assistant":
        messages.append({"role": "assistant", "content": str(prefix)})
    else:
        messages[-1] = {"role": "assistant", "content": str(prefix)}
    return messages


def _device(requested: str | None) -> torch.device:
    value = str(requested or os.getenv("POLYSEG_PRM_DEVICE", "auto")).strip().lower()
    if value in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_prm(model_path: str, device_name: str | None, max_length: int):
    """Load one frozen PRM per process/device, fixing old checkpoint metadata."""

    from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen2Model
    from safetensors import safe_open

    device = _device(device_name)
    key = (str(Path(model_path).expanduser().resolve()), str(device), int(max_length))
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Qwen2.5-Math-PRM800K's published config predates the pad token field;
        # the custom Qwen2 implementation accesses it during construction.
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 151643
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        # Use the maintained Qwen2 backbone; the published remote class is
        # incompatible with transformers 5.x (and is numerically unstable).
        backbone_config = Qwen2Config.from_pretrained(model_path)
        backbone_config.pad_token_id = int(getattr(config, "pad_token_id", 151643) or 151643)
        backbone = Qwen2Model.from_pretrained(model_path, config=backbone_config, torch_dtype=dtype)
        score = nn.Sequential(
            nn.Linear(backbone_config.hidden_size, backbone_config.hidden_size),
            nn.ReLU(),
            nn.Linear(backbone_config.hidden_size, 2),
        ).to(dtype=dtype)
        index_path = Path(model_path) / "model.safetensors.index.json"
        import json

        index = json.loads(index_path.read_text())
        shard = Path(model_path) / index["weight_map"]["score.0.weight"]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            score_state = score.state_dict()
            for local_name, remote_name in (
                ("0.weight", "score.0.weight"),
                ("0.bias", "score.0.bias"),
                ("2.weight", "score.2.weight"),
                ("2.bias", "score.2.bias"),
            ):
                score_state[local_name].copy_(handle.get_tensor(remote_name))
        model = _ModernPRM(backbone, score).eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _MODEL_CACHE[key] = (tokenizer, model, device)
        return tokenizer, model, device


def _separator_probability(tokenizer, model, device: torch.device, messages: list[dict[str, str]], max_length: int) -> float:
    """Return positive-class probability at the final ``<extra_0>`` token."""

    conversation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # A content string may escape the special token under a custom template;
    # tokenizing it explicitly guarantees one classifier endpoint.
    separator = "<extra_0>"
    if not conversation.endswith(separator):
        conversation = conversation + separator
    encoded = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    separator_id = tokenizer.convert_tokens_to_ids(separator)
    positions = torch.nonzero(input_ids[0] == int(separator_id), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("PRM input lost its <extra_0> endpoint after tokenization")
    endpoint = int(positions[-1])
    with torch.inference_mode():
        # The checkpoint's remote Qwen2 implementation targets transformers
        # 4.4x; disabling KV cache keeps it compatible with current cache APIs
        # while preserving the token-classification logits.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, endpoint].float()
        probability = F.softmax(logits, dim=-1)[1]
    return float(probability.clamp(0.0, 1.0).cpu())


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    prm_model_path: str | None = None,
    prm_device: str | None = None,
    prm_max_length: int = 4096,
    prm_query_cost: float = 1.0,
    **_: Any,
) -> dict[str, Any]:
    """Score one segment prefix with frozen PRM plus optional exact verifier."""

    extra_info = dict(extra_info or {})
    model_path = prm_model_path or os.getenv("POLYSEG_PRM_MODEL_PATH")
    if not model_path:
        raise RuntimeError("PRM feedback requires prm_model_path or POLYSEG_PRM_MODEL_PATH")
    prefix = str(extra_info.get("segment_prefix", solution_str))
    raw_prompt = extra_info.get("raw_prompt", "")
    tokenizer, model, device = _load_prm(model_path, prm_device, int(prm_max_length))
    utility = _separator_probability(
        tokenizer,
        model,
        device,
        _messages(raw_prompt, prefix),
        int(prm_max_length),
    )
    result: dict[str, Any] = {
        "utility": utility,
        "prm_score": utility,
        "model_derived_pseudo_label": True,
        "feedback_protocol_contract": feedback_protocol_contract("prm_potential"),
        "prm_model_path": str(model_path),
        "query_cost": float(prm_query_cost),
        "prm_segment_id": int(extra_info.get("segment_id", 0)),
    }
    # Keep the exact outcome metric separate from the PRM potential.  The
    # reward manager uses this field only for final_accuracy diagnostics.
    if int(extra_info.get("segment_id", 0)) == int(extra_info.get("segment_count", 1)) - 1:
        full_response = str(extra_info.get("full_response", solution_str))
        exact = default_compute_score(data_source, full_response, ground_truth, extra_info=extra_info)
        # math_dapo/aime verifiers return structured ``{score, acc, pred}``,
        # while GSM8K returns a scalar.  Keep the final outcome metric
        # separate from the PRM potential in both cases.
        if isinstance(exact, dict):
            result["acc"] = float(exact.get("acc", float(exact.get("score", 0.0)) > 0.0))
            result["verifier_score"] = float(exact.get("score", 0.0))
            if exact.get("pred") is not None:
                result["verifier_pred"] = exact["pred"]
        else:
            result["acc"] = float(exact > 0.5)
            result["verifier_score"] = float(exact)
    return result


__all__ = ["compute_score", "_messages", "_separator_probability"]
