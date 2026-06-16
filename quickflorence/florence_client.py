"""Florence2 MCP Server - Vision understanding tools via Model Context Protocol."""

import sys
import os
from typing import Optional, Union

# Force stderr logging for MCP stdio transport (stdout is reserved for JSON-RPC)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ─── Model Registry ──────────────────────────────────────────────

# Human-readable aliases → HuggingFace IDs. Default model selected via
# FLORENCE_MODEL env var (accepts alias or HF ID), falling back to large.
_MODELS: dict[str, str] = {
    "florence2-base": "microsoft/Florence-2-base",
    "florence2-large": "microsoft/Florence-2-large",
    "florence2-base-ft": "microsoft/Florence-2-base-ft",
    "florence2-large-ft": "microsoft/Florence-2-large-ft",
}

# Reverse lookup: HF ID → alias (for display)
_HF_TO_ALIAS: dict[str, str] = {v: k for k, v in _MODELS.items()}

_DEFAULT_MODEL_KEY = os.environ.get("FLORENCE_MODEL", "florence2-large")


def list_models() -> list[dict]:
    """Return metadata about all supported Florence-2 models."""
    return [
        {"alias": alias, "hf_id": hf_id} for alias, hf_id in _MODELS.items()
    ]


def _resolve_model_key(model_hint: str) -> str:
    """Resolve a model hint (alias or HF ID) to a canonical alias key.

    Falls back to the default if the hint is unrecognised.
    """
    # Direct alias match
    if model_hint in _MODELS:
        return model_hint
    # HF ID → alias
    alias = _HF_TO_ALIAS.get(model_hint)
    if alias:
        return alias
    print(
        f"[QuickFlorence] WARNING: Unknown model '{model_hint}'. "
        f"Supported aliases: {', '.join(sorted(_MODELS))}. "
        f"Falling back to default ('{_DEFAULT_MODEL_KEY}').",
        file=sys.stderr,
    )
    return _DEFAULT_MODEL_KEY


def _get_hf_id(model_key: str) -> str:
    """Get the HuggingFace model ID for a canonical alias key."""
    return _MODELS[model_key]


def _resolve_device(device_hint: Optional[str] = None) -> str:
    """Resolve the target device for model inference.

    Priority order:
    1. Explicit hint (QUICKFLORENCE_DEVICE env var or CLI --device)
    2. Auto-detect: CUDA > ROCm > CPU

    Supported values: 'cuda', 'cuda:N', 'rocm', 'cpu'
    """
    import torch

    # Normalize the hint
    if device_hint:
        device_hint = device_hint.strip().lower()

    # If explicit device requested, validate it
    if device_hint in ("cuda", "cuda:0"):
        if not torch.cuda.is_available():
            print(
                "[QuickFlorence] WARNING: CUDA requested but not available. "
                "Falling back to auto-detect.",
                file=sys.stderr,
            )
        else:
            return device_hint if device_hint == "cuda" else "cuda:0"
    elif device_hint and device_hint.startswith("cuda:"):
        try:
            idx = int(device_hint.split(":")[1])
            if torch.cuda.is_available() and 0 <= idx < torch.cuda.device_count():
                return device_hint
            print(
                f"[QuickFlorence] WARNING: CUDA:{idx} requested but not available "
                f"(GPU count: {torch.cuda.device_count()}). Falling back to auto-detect.",
                file=sys.stderr,
            )
        except (ValueError, IndexError):
            print(
                f"[QuickFlorence] WARNING: Invalid device '{device_hint}'. Falling back to auto-detect.",
                file=sys.stderr,
            )
    elif device_hint == "rocm":
        # ROCm is exposed via torch.cuda with HIP backend
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return "cuda"  # PyTorch uses 'cuda' device string for ROCm too
        print(
            "[QuickFlorence] WARNING: ROCm requested but PyTorch was not built with HIP support. "
            "Install ROCm PyTorch: pip install torch --index-url https://download.pytorch.org/whl/rocm6.x",
            file=sys.stderr,
        )
    elif device_hint == "cpu":
        return "cpu"

    # Auto-detect: CUDA > ROCm > CPU
    if torch.cuda.is_available():
        backend = "ROCm (HIP)" if hasattr(torch.version, "hip") and torch.version.hip is not None else "CUDA"
        print(f"[QuickFlorence] Auto-detected {backend} GPU available.", file=sys.stderr)
        return "cuda"

    return "cpu"


# Per-model cache: {model_key: (model, processor)}
_model_cache: dict[str, tuple] = {}

# Shared device (all models run on the same device)
_device: Optional[object] = None


def _ensure_loaded(
    model_hint: Optional[str] = None,
    device_hint: Optional[str] = None,
):
    """Lazily load a Florence2 model and processor by alias key.

    Models are cached per-key so switching between base/large doesn't
    require re-downloading — both stay in RAM/GPU memory.

    Args:
        model_hint: Alias (e.g. 'florence2-base') or HF ID.  Falls back to
                    FLORENCE_MODEL env var, then 'florence2-large'.
        device_hint: Optional device override (e.g., 'cuda', 'rocm', 'cpu').
                     If not provided, reads QUICKFLORENCE_DEVICE env var or auto-detects.
    """
    global _device

    model_key = _resolve_model_key(model_hint or _DEFAULT_MODEL_KEY)

    # Already cached? Return immediately.
    if model_key in _model_cache:
        return _model_cache[model_key]

    # Resolve device once (shared across all models)
    if _device is None:
        effective_hint = device_hint or os.environ.get("QUICKFLORENCE_DEVICE")
        resolved_device = _resolve_device(effective_hint)
        import torch
        _device = torch.device(resolved_device)

    hf_id = _get_hf_id(model_key)
    print(f"Loading {model_key} ({hf_id}) on {_device}...", file=sys.stderr)

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    # Use half-precision on GPU for memory efficiency
    dtype = (
        torch.float16 if "cuda" in str(_device) else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(_device)

    processor = AutoProcessor.from_pretrained(
        hf_id,
        trust_remote_code=True,
    )

    _model_cache[model_key] = (model, processor)
    print(f"{model_key} loaded successfully.", file=sys.stderr)
    return model, processor


def run_inference(
    image_path: str,
    task: str,
    text_input: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Run Florence2 inference on an image.

    Args:
        image_path: Path to the image file.
        task: Task prompt (e.g., "<CAPTION>", "<OD>").
        text_input: Optional text input for tasks that support it.
        model: Model alias key (e.g. 'florence2-base') or HF ID.  Falls back
               to FLORENCE_MODEL env var, then 'florence2-large'.

    Returns:
        Dictionary with parsed results from Florence2.
    """
    model_obj, processor = _ensure_loaded(model_hint=model)

    from PIL import Image

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    # Build prompt
    prompt = task + text_input if text_input else task

    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(_device)

    # Generate
    generated_ids = model_obj.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )

    return parsed_answer
