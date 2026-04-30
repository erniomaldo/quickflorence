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


def _ensure_loaded():
    """Lazily load Florence2 model and processor on first call."""
    global _model, _processor, _device

    if _model is not None and _processor is not None:
        return

    print("Loading Florence-2-large model...", file=sys.stderr)

    global _torch
    import torch

    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}", file=sys.stderr)

    _model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
    ).to(_device)

    _processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
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

