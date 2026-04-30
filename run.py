#!/usr/bin/env python3
"""QuickFlorence MCP Server entry point."""
import sys
import os

# Redirect HF cache to user-writable directory (root owns ~/.cache/huggingface)
HF_CACHE = "/home/ernesto-abec/.hf-cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")

# Ensure the parent of 'quickflorence/' is on the path
sys.path.insert(0, "/home/ernesto-abec/MCP")

from quickflorence.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
