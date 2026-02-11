"""
RunPod Serverless Handler for Flux2 Image Generation
Supports Flux Dev and Flux Schnell models with full feature set
"""
import runpod
import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO
from PIL import Image
import os

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "black-forest-labs/FLUX.1-dev")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading Flux model: {MODEL_NAME}")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# Load the pipeline globally (persists across requests)
pipe = FluxPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE
).to(DEVICE)

# Enable memory optimizations
if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    # Optional: Enable CPU offload for very large models
    # pipe.enable_sequential_cpu_offload()

print("Flux model loaded successfully!")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def handler(event):
    """
    Handler for Flux2 image generation.

    Input schema:
    {
        "input": {
            "prompt": str (required) - Text prompt for image generation
            "negative_prompt": str (optional) - Negative prompt
            "width": int (optional, default: 1024) - Image width
            "height": int (optional, default: 1024) - Image height
            "num_inference_steps": int (optional, default: 50 for dev, 4 for schnell)
            "guidance_scale": float (optional, default: 7.5) - CFG scale
            "num_images": int (optional, default: 1) - Number of images to generate
            "seed": int (optional) - Random seed for reproducibility
            "output_format": str (optional, default: "PNG") - Output format (PNG, JPEG, WEBP)
            "output_quality": int (optional, default: 95) - JPEG quality (1-100)
        }
    }

    Returns:
    {
        "status": "success" | "error",
        "images": [base64_encoded_image_1, ...],
        "metadata": {
            "prompt": str,
            "seed": int,
            "inference_time": float
        }
    }
    """
    import time
    start_time = time.time()

    try:
        # Extract input parameters
        job_input = event.get("input", {})

        # Required parameters
        prompt = job_input.get("prompt")
        if not prompt:
            return {
                "status": "error",
                "error": "Missing required parameter: prompt"
            }

        # Optional parameters with defaults
        negative_prompt = job_input.get("negative_prompt", "")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)

        # Default steps depend on model variant
        default_steps = 4 if "schnell" in MODEL_NAME.lower() else 50
        num_inference_steps = job_input.get("num_inference_steps", default_steps)

        guidance_scale = job_input.get("guidance_scale", 7.5)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")
        output_format = job_input.get("output_format", "PNG").upper()
        output_quality = job_input.get("output_quality", 95)

        # Validate parameters
        if width % 8 != 0 or height % 8 != 0:
            return {
                "status": "error",
                "error": "Width and height must be multiples of 8"
            }

        if num_images < 1 or num_images > 4:
            return {
                "status": "error",
                "error": "num_images must be between 1 and 4"
            }

        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        else:
            # Generate random seed for reproducibility tracking
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        print(f"Generating {num_images} image(s) with prompt: '{prompt[:50]}...'")
        print(f"Parameters: {width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}, seed={seed}")

        # Generate images
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator
            )

        # Convert images to base64
        images_b64 = []
        for img in result.images:
            img_b64 = image_to_base64(img, format=output_format)
            images_b64.append(img_b64)

        inference_time = time.time() - start_time

        return {
            "status": "success",
            "images": images_b64,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "inference_time": round(inference_time, 2),
                "model": MODEL_NAME
            }
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error occurred: {error_trace}")

        return {
            "status": "error",
            "error": str(e),
            "traceback": error_trace
        }


if __name__ == "__main__":
    # Start the serverless function
    runpod.serverless.start({"handler": handler})
