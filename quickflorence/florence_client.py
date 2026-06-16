"""Florence2 MCP Server - Vision understanding tools via Model Context Protocol."""

import sys
import os
from typing import Optional, Union

# Force stderr logging for MCP stdio transport (stdout is reserved for JSON-RPC)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Module-level cache for lazy loading
_model = None
_processor = None
_device = None

# Supported Florence-2 model identifiers
SUPPORTED_MODELS = frozenset({
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft",
})

# Configurable model via FLORENCE_MODEL env var (default: large)
MODEL_NAME = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")


def _validate_model(model_name: str) -> str:
    """Validate and return the effective Florence-2 model name.

    Falls back to default if the requested model is not in the supported list.
    """
    if model_name in SUPPORTED_MODELS:
        return model_name
    print(
        f"[QuickFlorence] WARNING: Unknown model '{model_name}'. "
        f"Supported models: {', '.join(sorted(SUPPORTED_MODELS))}. "
        f"Falling back to default.",
        file=sys.stderr,
    )
    return "microsoft/Florence-2-large"


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


def _ensure_loaded(device_hint: Optional[str] = None):
    """Lazily load Florence2 model and processor on first call.

    Args:
        device_hint: Optional device override (e.g., 'cuda', 'rocm', 'cpu').
                     If not provided, reads QUICKFLORENCE_DEVICE env var or auto-detects.
    """
    global _model, _processor, _device

    if _model is not None and _processor is not None:
        return

    # Resolve device from hint > env var > auto-detect
    effective_hint = device_hint or os.environ.get("QUICKFLORENCE_DEVICE")
    resolved_device = _resolve_device(effective_hint)

    # Validate and resolve model name
    effective_model = _validate_model(MODEL_NAME)
    print(f"Loading {effective_model}...", file=sys.stderr)

    global _torch
    import torch

    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    _device = torch.device(resolved_device)
    print(f"Using device: {_device}", file=sys.stderr)

    # Use half-precision on GPU for memory efficiency
    dtype = torch.float16 if "cuda" in resolved_device or resolved_device == "rocm" else torch.float32
    print(f"Using dtype: {dtype}", file=sys.stderr)

    _model = AutoModelForCausalLM.from_pretrained(
        effective_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(_device)

    _processor = AutoProcessor.from_pretrained(
        effective_model,
        trust_remote_code=True,
    )

    print("Model loaded successfully.", file=sys.stderr)


def run_inference(image_path: str, task: str, text_input: Optional[str] = None) -> dict:
    """
    Run Florence2 inference on an image.

    Args:
        image_path: Path to the image file.
        task: Task prompt (e.g., "<CAPTION>", "<OD>").
        text_input: Optional text input for tasks that support it.

    Returns:
        Dictionary with parsed results from Florence2.
    """
    _ensure_loaded()

    from PIL import Image

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    # Build prompt
    prompt = task + text_input if text_input else task

    # Prepare inputs
    inputs = _processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(_device)

    # Generate
    generated_ids = _model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Decode
    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process
    parsed_answer = _processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )

    return parsed_answer
