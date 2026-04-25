"""
scripts/serve.py
-----------------
Launch the DocuSeek FastAPI server with a given experiment config.

Sets DOCUSEEK_CONFIG_PATH so the lifespan in main.py knows which pipeline
to load, then hands off to uvicorn.

Usage
-----
    uv run python scripts/serve.py --config experiments/05_generation/few_shot_cot/config.yaml
    uv run python scripts/serve.py --config experiments/05_generation/few_shot_cot/config.yaml --reload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the DocuSeek API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  uv run python scripts/serve.py --config experiments/00_baseline/config.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the experiment config YAML to serve.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.config.exists():
        print(f"[error] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Resolve to absolute path — uvicorn changes cwd in some reload modes
    os.environ["DOCUSEEK_CONFIG_PATH"] = str(args.config.resolve())

    uvicorn.run(
        "docuseek.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
