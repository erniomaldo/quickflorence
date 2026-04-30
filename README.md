# QuickFlorence

MCP Server for Florence-2 Large — a vision model for image understanding tasks including captioning, object detection, OCR, segmentation, and more.

## Prerequisites: Install `uv`

QuickFlorence uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for fast package installation and execution via `uvx`.

Install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternative installation methods (pip, Homebrew, etc.) are available at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

## Installation & First Run

**Important:** Run the following command in your terminal **before** configuring QuickFlorence in your IDE:

```bash
uvx --from git+https://github.com/erniomaldo/quickflorence quickflorence
```

This first run is necessary because:

- The Florence-2-large model (~several GB) downloads automatically to `~/.cache/huggingface`
- Dependencies (torch, transformers, etc.) are installed
- Initial model loading takes ~30+ seconds
- If you skip this step, your IDE's MCP connection timeout may fail while the model downloads/loads

After the first successful run, the model is cached and subsequent starts are much faster.

## IDE Configuration (KiloCode / OpenCode)

Add the following to your `opencode.jsonc` (or equivalent MCP config):

```json
"quickflorence": {
  "type": "local",
  "command": [
    "uvx",
    "--from",
    "git+https://github.com/erniomaldo/quickflorence",
    "quickflorence"
  ],
  "enabled": true
}
```

## Available MCP Tools

| Tool | Description | Parameters |
|---|---|---|
| `caption` | Generate a text caption describing an image | `image_path` (str), `detail_level` (str: `simple`, `detailed`, `more_detailed`) |
| `detect_objects` | Detect objects with bounding boxes and labels | `image_path` (str) |
| `dense_region_caption` | Generate captions for every region with bounding boxes | `image_path` (str) |
| `phrase_grounding` | Find and localize a specific phrase within an image | `image_path` (str), `phrase` (str) |
| `segment_by_expression` | Segment regions matching a referring expression | `image_path` (str), `expression` (str) |
| `ocr` | Extract text from an image (OCR) | `image_path` (str), `with_regions` (bool, default: `false`) |
| `analyze_image` | Generic Florence2 inference for any supported task | `image_path` (str), `task_mode` (str), `text_input` (str, optional) |

## Supported Task Modes

The `analyze_image` tool accepts any of these Florence-2 task modes:

| Task Mode | Description |
|---|---|
| `<CAPTION>` | Brief one-line caption |
| `<DETAILED_CAPTION>` | Detailed caption including background elements |
| `<MORE_DETAILED_CAPTION>` | Very thorough caption with fine details |
| `<OD>` | Object Detection — bounding boxes with labels |
| `<DENSE_REGION_CAPTION>` | Region-level captions with bounding boxes |
| `<CAPTION_TO_PHRASE_GROUNDING>` | Ground a text phrase to bounding boxes |
| `<REFERRING_EXPRESSION_SEGMENTATION>` | Segment regions matching an expression |
| `<OCR>` | Extract text without coordinates |
| `<OCR_WITH_REGION>` | Extract text with bounding box coordinates |

## System Requirements

- **`uv`** (includes `uvx`) — Python package installer and runner. See [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Python >= 3.10**
- **GPU recommended** (CUDA) — CPU fallback works but is slower
- **Disk space** — several GB for the model cache at `~/.cache/huggingface`

## Project Structure

```
quickflorence/
├── pyproject.toml
├── requirements.txt
└── quickflorence/
    ├── __init__.py
    ├── server.py
    └── florence_client.py
```

## Important Notes

- The model downloads automatically on first run
- HuggingFace cache is stored at `~/.cache/huggingface`
- Uses stdio transport (MCP compatible)
- Automatically detects and uses CUDA if available, falls back to CPU
