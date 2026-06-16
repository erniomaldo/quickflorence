"""QuickFlorence MCP Server - Florence2 vision tools exposed via Model Context Protocol."""

import sys
import os
import argparse
from typing import Optional

# Redirect HF cache to user-writable directory
hf_cache = os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf_cache, exist_ok=True)
os.environ["HF_HOME"] = hf_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache, "hub")

from mcp.server.fastmcp import FastMCP

from .florence_client import run_inference, _ensure_loaded, list_models, _MODELS
from .ui_lm_studio import (
    detect_ui_elements as _detect_ui,
    describe_screen as _describe_screen,
    find_ui_element as _find_ui_element,
    check_lm_studio_health,
)


# ─── CLI Args & Device Hint ──────────────────────────────────────

_DEVICE_HINT: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QuickFlorence MCP Server - Florence2 vision model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: 'cuda', 'cuda:N', 'rocm', or 'cpu'. "
             "Defaults to QUICKFLORENCE_DEVICE env var or auto-detect (CUDA > ROCm > CPU).",
    )
    return parser.parse_args()


# ─── MCP Server ──────────────────────────────────────────────────

def _init_mcp():
    """Initialize the FastMCP server with all tools."""
    mcp = FastMCP(
        "quickflorence",
        instructions="Florence-2 large vision model for image understanding tasks including captioning, object detection, OCR, segmentation, and more.",
    )

    # ─── Model Discovery Tool ────────────────────────────────

    @mcp.tool()
    def list_florence_models() -> str:
        """List all available Florence-2 models with their aliases and HuggingFace IDs.

        Use the alias (e.g., 'florence2-base') in other tools via the `model` parameter.
        """
        import json
        models = list_models()
        lines = [f"Available Florence-2 models ({len(models)}):"]
        for m in models:
            lines.append(f"  - {m['alias']}: {m['hf_id']}")
        return "\n".join(lines)

    # ─── Captioning Tools ──────────────────────────────────────

    @mcp.tool()
    def caption(image_path: str, detail_level: str = "simple", model: Optional[str] = None) -> str:
        """Generate a text caption describing the contents of an image.

        Args:
            image_path: Absolute path to the image file to analyze.
            detail_level: Level of detail for the caption. One of: 'simple', 'detailed', 'more_detailed'.
                - simple: Brief one-line description.
                - detailed: More thorough description including background elements.
                - more_detailed: Very thorough description with fine details.
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        task_map = {
            "simple": "<CAPTION>",
            "detailed": "<DETAILED_CAPTION>",
            "more_detailed": "<MORE_DETAILED_CAPTION>",
        }

        task = task_map.get(detail_level, "<CAPTION>")
        result = run_inference(image_path, task, model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        caption_key = list(result.keys())[0]
        return str(result[caption_key])

    # ─── Object Detection Tools ────────────────────────────────

    @mcp.tool()
    def detect_objects(image_path: str, model: Optional[str] = None) -> str:
        """Detect objects in an image and return bounding boxes with labels.

        Args:
            image_path: Absolute path to the image file to analyze.
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        result = run_inference(image_path, "<OD>", model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        data = result["<OD>"]
        lines = []
        for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
            x1, y1, x2, y2 = [int(c) for c in bbox]
            lines.append(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")

        return f"Detected {len(data.get('labels', []))} object(s):\n" + "\n".join(lines) if lines else "No objects detected."

    @mcp.tool()
    def dense_region_caption(image_path: str, model: Optional[str] = None) -> str:
        """Generate detailed captions for every region in the image with bounding boxes.

        Args:
            image_path: Absolute path to the image file to analyze.
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        result = run_inference(image_path, "<DENSE_REGION_CAPTION>", model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        data = result["<DENSE_REGION_CAPTION>"]
        lines = []
        for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
            x1, y1, x2, y2 = [int(c) for c in bbox]
            lines.append(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")

        return f"Found {len(data.get('labels', []))} region(s):\n" + "\n".join(lines) if lines else "No regions found."

    # ─── Grounding Tools ──────────────────────────────────────

    @mcp.tool()
    def phrase_grounding(image_path: str, phrase: str, model: Optional[str] = None) -> str:
        """Find and localize a specific phrase/description within an image.

        Args:
            image_path: Absolute path to the image file to analyze.
            phrase: The phrase or description to find in the image (e.g., 'a wine bottle').
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        result = run_inference(image_path, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase, model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        data = result["<CAPTION_TO_PHRASE_GROUNDING>"]
        lines = []
        for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
            x1, y1, x2, y2 = [int(c) for c in bbox]
            lines.append(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")

        if lines:
            return f"Found '{phrase}' at:\n" + "\n".join(lines)
        return f"'{phrase}' not found in the image."

    @mcp.tool()
    def segment_by_expression(image_path: str, expression: str, model: Optional[str] = None) -> str:
        """Segment regions of an image that match a referring expression.

        Returns polygon coordinates and labels for matching regions.

        Args:
            image_path: Absolute path to the image file to analyze.
            expression: Referring expression to segment (e.g., 'a wine bottle').
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        result = run_inference(image_path, "<REFERRING_EXPRESSION_SEGMENTATION>", text_input=expression, model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        data = result["<REFERRING_EXPRESSION_SEGMENTATION>"]
        lines = []
        for polygons, label in zip(data.get("polygons", []), data.get("labels", [])):
            points_count = sum(len(p) // 2 for p in polygons) if polygons else 0
            lines.append(f"  - {label}: polygon with {points_count} points")

        return f"Segmented '{expression}' into {len(data.get('labels', []))} region(s):\n" + "\n".join(lines) if lines else "No matching regions found."

    # ─── OCR Tools ─────────────────────────────────────────────

    @mcp.tool()
    def ocr(image_path: str, with_regions: bool = False, model: Optional[str] = None) -> str:
        """Extract text from an image using Optical Character Recognition.

        Args:
            image_path: Absolute path to the image file to analyze.
            with_regions: If True, also return bounding box coordinates for each text region.
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        task = "<OCR_WITH_REGION>" if with_regions else "<OCR>"
        result = run_inference(image_path, task, model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        data = result[task]

        if with_regions and isinstance(data, dict):
            lines = []
            for box, label in zip(data.get("quad_boxes", []), data.get("labels", [])):
                corners = [f"({int(box[i])}, {int(box[i+1])})" for i in range(0, len(box), 2)]
                lines.append(f'  - "{label}": corners={corners}')
            return f"Extracted {len(data.get('labels', []))} text region(s):\n" + "\n".join(lines)

        return str(data)

    # ─── Generic Tool (catch-all for any Florence2 task) ──────

    @mcp.tool()
    def analyze_image(image_path: str, task_mode: str, text_input: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generic Florence2 inference for any supported task mode.

        Supported task modes: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>,
            <OD>, <DENSE_REGION_CAPTION>, <CAPTION_TO_PHRASE_GROUNDING>,
            <REFERRING_EXPRESSION_SEGMENTATION>, <OCR>, <OCR_WITH_REGION>

        Args:
            image_path: Absolute path to the image file to analyze.
            task_mode: Florence2 task prompt (e.g., '<CAPTION>', '<OD>').
            text_input: Optional text input for tasks that accept it (grounding, segmentation).
            model: Florence-2 model alias (e.g., 'florence2-base' for speed,
                'florence2-large' for precision). Defaults to FLORENCE_MODEL env var or large.
        """
        result = run_inference(image_path, task_mode, text_input=text_input, model=model)

        if "error" in result:
            return f"Error: {result['error']}"

        import json
        return json.dumps(result, indent=2, default=str)

    # ─── UI-TARS Tools (via LM Studio) ────────────────────────

    @mcp.tool()
    def detect_ui_elements(image_path: str) -> str:
        """Detect all interactive UI elements in a screenshot using UI-TARS.

        Returns a structured list of buttons, inputs, links, labels and their
        approximate positions. Useful for GUI automation and testing.

        Args:
            image_path: Absolute path to the screenshot image file.
        """
        return _detect_ui(image_path)

    @mcp.tool()
    def describe_screen(image_path: str, detail: str = "normal") -> str:
        """Generate a natural-language description of what's shown on a screen.

        Uses UI-TARS (via LM Studio) to provide GUI-aware descriptions useful
        for accessibility, documentation, or automated testing.

        Args:
            image_path: Absolute path to the screenshot image file.
            detail: Level of detail. One of: 'brief', 'normal', 'detailed'.
        """
        return _describe_screen(image_path, detail)

    @mcp.tool()
    def find_ui_element(image_path: str, description: str) -> str:
        """Find a specific UI element on a screenshot by natural language description.

        Locates elements like 'the login button', 'the search input', or 'the profile icon'
        and reports their position, type, and interactivity.

        Args:
            image_path: Absolute path to the screenshot image file.
            description: Natural language description of the element to find.
        """
        return _find_ui_element(image_path, description)

    @mcp.tool()
    def ui_tars_health() -> str:
        """Check if LM Studio (UI-TARS backend) is reachable and the model is loaded.

        Returns the health status of the LM Studio connection and model availability.
        """
        return check_lm_studio_health()

    return mcp


# ─── Main Entry Point ──────────────────────────────────────────────

def main():
    """Run the MCP server."""
    global _DEVICE_HINT

    args = parse_args()
    if args.device:
        _DEVICE_HINT = args.device
        print(f"[QuickFlorence] Device set via CLI: {_DEVICE_HINT}", file=sys.stderr)

    # Pre-warm model with device hint (lazy load on first tool call)
    _ensure_loaded(_DEVICE_HINT)

    mcp = _init_mcp()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
