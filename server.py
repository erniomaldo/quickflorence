"""QuickFlorence MCP Server - Florence2 vision tools exposed via Model Context Protocol."""

import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .florence_client import run_inference

# Initialize FastMCP server
mcp = FastMCP(
    "quickflorence",
    instructions="Florence-2 large vision model for image understanding tasks including captioning, object detection, OCR, segmentation, and more.",
)


# ─── Captioning Tools ──────────────────────────────────────────────

@mcp.tool()
def caption(image_path: str, detail_level: str = "simple") -> str:
    """Generate a text caption describing the contents of an image.

    Args:
        image_path: Absolute path to the image file to analyze.
        detail_level: Level of detail for the caption. One of: 'simple', 'detailed', 'more_detailed'.
            - simple: Brief one-line description.
            - detailed: More thorough description including background elements.
            - more_detailed: Very thorough description with fine details.
    """
    task_map = {
        "simple": "<CAPTION>",
        "detailed": "<DETAILED_CAPTION>",
        "more_detailed": "<MORE_DETAILED_CAPTION>",
    }

    task = task_map.get(detail_level, "<CAPTION>")
    result = run_inference(image_path, task)

    if "error" in result:
        return f"Error: {result['error']}"

    # Extract caption text from result dict
    caption_key = list(result.keys())[0]
    return str(result[caption_key])


# ─── Object Detection Tools ────────────────────────────────────────

@mcp.tool()
def detect_objects(image_path: str) -> str:
    """Detect objects in an image and return bounding boxes with labels.

    Args:
        image_path: Absolute path to the image file to analyze.
    """
    result = run_inference(image_path, "<OD>")

    if "error" in result:
        return f"Error: {result['error']}"

    data = result["<OD>"]
    lines = []
    for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        lines.append(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")

    return f"Detected {len(data.get('labels', []))} object(s):\n" + "\n".join(lines) if lines else "No objects detected."


@mcp.tool()
def dense_region_caption(image_path: str) -> str:
    """Generate detailed captions for every region in the image with bounding boxes.

    Args:
        image_path: Absolute path to the image file to analyze.
    """
    result = run_inference(image_path, "<DENSE_REGION_CAPTION>")

    if "error" in result:
        return f"Error: {result['error']}"

    data = result["<DENSE_REGION_CAPTION>"]
    lines = []
    for bbox, label in zip(data.get("bboxes", []), data.get("labels", [])):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        lines.append(f"  - {label}: [{x1}, {y1}, {x2}, {y2}]")

    return f"Found {len(data.get('labels', []))} region(s):\n" + "\n".join(lines) if lines else "No regions found."


# ─── Grounding Tools ──────────────────────────────────────────────

@mcp.tool()
def phrase_grounding(image_path: str, phrase: str) -> str:
    """Find and localize a specific phrase/description within an image.

    Args:
        image_path: Absolute path to the image file to analyze.
        phrase: The phrase or description to find in the image (e.g., 'a wine bottle').
    """
    result = run_inference(image_path, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase)

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
def segment_by_expression(image_path: str, expression: str) -> str:
    """Segment regions of an image that match a referring expression.

    Returns polygon coordinates and labels for matching regions.

    Args:
        image_path: Absolute path to the image file to analyze.
        expression: Referring expression to segment (e.g., 'a wine bottle').
    """
    result = run_inference(image_path, "<REFERRING_EXPRESSION_SEGMENTATION>", text_input=expression)

    if "error" in result:
        return f"Error: {result['error']}"

    data = result["<REFERRING_EXPRESSION_SEGMENTATION>"]
    lines = []
    for polygons, label in zip(data.get("polygons", []), data.get("labels", [])):
        points_count = sum(len(p) // 2 for p in polygons) if polygons else 0
        lines.append(f"  - {label}: polygon with {points_count} points")

    return f"Segmented '{expression}' into {len(data.get('labels', []))} region(s):\n" + "\n".join(lines) if lines else "No matching regions found."


# ─── OCR Tools ─────────────────────────────────────────────────────

@mcp.tool()
def ocr(image_path: str, with_regions: bool = False) -> str:
    """Extract text from an image using Optical Character Recognition.

    Args:
        image_path: Absolute path to the image file to analyze.
        with_regions: If True, also return bounding box coordinates for each text region.
    """
    task = "<OCR_WITH_REGION>" if with_regions else "<OCR>"
    result = run_inference(image_path, task)

    if "error" in result:
        return f"Error: {result['error']}"

    data = result[task]

    if with_regions and isinstance(data, dict):
        lines = []
        for box, label in zip(data.get("quad_boxes", []), data.get("labels", [])):
            # Format quad box as 4 corner points
            corners = [f"({int(box[i])}, {int(box[i+1])})" for i in range(0, len(box), 2)]
            lines.append(f"  - \"{label}\": corners={corners}")
        return f"Extracted {len(data.get('labels', []))} text region(s):\n" + "\n".join(lines)

    # Plain OCR returns a string
    return str(data)


# ─── Generic Tool (catch-all for any Florence2 task) ──────────────

@mcp.tool()
def analyze_image(image_path: str, task_mode: str, text_input: Optional[str] = None) -> str:
    """Generic Florence2 inference for any supported task mode.

    Supported task modes: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>,
        <OD>, <DENSE_REGION_CAPTION>, <CAPTION_TO_PHRASE_GROUNDING>,
        <REFERRING_EXPRESSION_SEGMENTATION>, <OCR>, <OCR_WITH_REGION>

    Args:
        image_path: Absolute path to the image file to analyze.
        task_mode: Florence2 task prompt (e.g., '<CAPTION>', '<OD>').
        text_input: Optional text input for tasks that accept it (grounding, segmentation).
    """
    result = run_inference(image_path, task_mode, text_input=text_input)

    if "error" in result:
        return f"Error: {result['error']}"

    import json
    return json.dumps(result, indent=2, default=str)


# ─── Main Entry Point ──────────────────────────────────────────────

def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
