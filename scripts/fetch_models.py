#!/usr/bin/env python3
"""
Fetch models for Voice Scam Shield.

Currently supports downloading an AASIST TorchScript checkpoint.

Usage examples:
  python scripts/fetch_models.py --aasist-url https://example.com/aasist_scripted.pt \
      --output backend/models/aasist_scripted.pt

If you have a private model on Hugging Face, first `huggingface-cli login`, then
use `huggingface_hub` in a custom script or place the file manually.
"""

import argparse
import os
import sys
import urllib.request


def download(url: str, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"Downloading {url} -> {output}")
    urllib.request.urlretrieve(url, output)
    size = os.path.getsize(output)
    print(f"Done. {size/1e6:.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aasist-url", type=str, help="URL to TorchScript AASIST checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        default="backend/models/aasist_scripted.pt",
        help="Output path for the checkpoint",
    )
    args = parser.parse_args()

    if not args.aasist_url:
        print("--aasist-url is required. Provide a direct download URL to a TorchScript AASIST model.")
        return 2

    try:
        download(args.aasist_url, args.output)
    except Exception as e:
        print(f"Failed to download: {e}")
        return 1
    print("SUCCESS: Set AASIST_CHECKPOINT_PATH to", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


