"""
Model download script for pre-caching Flux models
Run this to download models before building Docker image
"""
import os
import argparse
from diffusers import FluxPipeline


def download_model(model_name: str, cache_dir: str = "./models"):
    """
    Download and cache Flux model

    Args:
        model_name: Model identifier (e.g., 'black-forest-labs/FLUX.1-dev')
        cache_dir: Directory to cache the model
    """
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")

    # Set cache directory
    os.environ['HF_HOME'] = cache_dir

    # Download model
    print("Starting download... This may take several minutes.")
    pipe = FluxPipeline.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    print(f"✅ Model downloaded successfully to {cache_dir}")
    print(f"Model size: {get_dir_size(cache_dir) / (1024**3):.2f} GB")


def get_dir_size(path: str) -> int:
    """Get total size of directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Flux model")
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Model name (default: FLUX.1-schnell)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Cache directory (default: ./models)"
    )

    args = parser.parse_args()

    download_model(args.model, args.cache_dir)
