"""
RunPod Serverless Handler for Flux2 - ALL WORKFLOWS
Supports: Text-to-Image, Image-to-Image, and Multi-Reference
Compatible with both FLUX.1 (dual encoder) and FLUX.2 (single encoder)
"""
import runpod
import torch
from diffusers import DiffusionPipeline, FluxImg2ImgPipeline, FluxControlNetPipeline
import base64
from io import BytesIO
from PIL import Image
import os

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "black-forest-labs/FLUX.2-dev")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading Flux models: {MODEL_NAME}")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# Load pipelines globally (persists across requests)
# Use DiffusionPipeline for auto-detection of pipeline type
txt2img_pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE
).to(DEVICE)

# Check if model has dual encoders (FLUX.1) or single encoder (FLUX.2)
has_dual_encoders = hasattr(txt2img_pipe, 'text_encoder_2')

print(f"Model architecture: {'Dual encoder (FLUX.1)' if has_dual_encoders else 'Single encoder (FLUX.2)'}")

# Build img2img pipeline with appropriate components
img2img_components = {
    'transformer': txt2img_pipe.transformer,
    'scheduler': txt2img_pipe.scheduler,
    'vae': txt2img_pipe.vae,
    'text_encoder': txt2img_pipe.text_encoder,
    'tokenizer': txt2img_pipe.tokenizer,
}

# Add dual encoder components if available (FLUX.1)
if has_dual_encoders:
    img2img_components['text_encoder_2'] = txt2img_pipe.text_encoder_2
    img2img_components['tokenizer_2'] = txt2img_pipe.tokenizer_2

# Create Image-to-Image pipeline (shares components with txt2img)
img2img_pipe = FluxImg2ImgPipeline(**img2img_components).to(DEVICE)

# Enable memory optimizations
if DEVICE == "cuda":
    txt2img_pipe.enable_attention_slicing()
    img2img_pipe.enable_attention_slicing()

print("Flux models loaded successfully!")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(img_data))
    return image.convert('RGB')


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def detect_workflow(job_input: dict) -> str:
    """
    Detect which workflow to use based on input parameters

    Returns: 'txt2img', 'img2img', or 'multi_reference'
    """
    has_init_image = job_input.get("init_image") is not None
    has_reference_images = job_input.get("reference_images") is not None

    if has_reference_images:
        return "multi_reference"
    elif has_init_image:
        return "img2img"
    else:
        return "txt2img"


def handler(event):
    """
    Universal handler for all Flux2 workflows.

    Input schema:
    {
        "input": {
            // Common parameters
            "prompt": str (required) - Text prompt for image generation
            "negative_prompt": str (optional) - Negative prompt
            "width": int (optional, default: 1024) - Image width
            "height": int (optional, default: 1024) - Image height
            "num_inference_steps": int (optional, default: 50 for dev, 4 for schnell)
            "guidance_scale": float (optional, default: 7.5) - CFG scale
            "num_images": int (optional, default: 1) - Number of images to generate
            "seed": int (optional) - Random seed for reproducibility
            "output_format": str (optional, default: "PNG") - Output format

            // Image-to-Image specific
            "init_image": str (optional) - Base64 encoded initial image
            "strength": float (optional, default: 0.8) - How much to transform (0.0-1.0)

            // Multi-Reference specific
            "reference_images": list[str] (optional) - List of base64 encoded reference images
            "reference_weights": list[float] (optional) - Weight for each reference (default: equal)
        }
    }

    Returns:
    {
        "status": "success" | "error",
        "workflow": "txt2img" | "img2img" | "multi_reference",
        "images": [base64_encoded_image_1, ...],
        "metadata": {...}
    }
    """
    import time
    start_time = time.time()

    try:
        job_input = event.get("input", {})

        # Required parameters
        prompt = job_input.get("prompt")
        if not prompt:
            return {
                "status": "error",
                "error": "Missing required parameter: prompt"
            }

        # Detect workflow
        workflow = detect_workflow(job_input)
        print(f"Detected workflow: {workflow}")

        # Common parameters
        negative_prompt = job_input.get("negative_prompt", "")
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        default_steps = 4 if "schnell" in MODEL_NAME.lower() else 50
        num_inference_steps = job_input.get("num_inference_steps", default_steps)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")
        output_format = job_input.get("output_format", "PNG").upper()

        # Validate common parameters
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

        # Set random seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Execute appropriate workflow
        if workflow == "txt2img":
            result = run_txt2img(
                prompt, negative_prompt, width, height,
                num_inference_steps, guidance_scale, num_images, generator
            )

        elif workflow == "img2img":
            strength = job_input.get("strength", 0.8)
            init_image_b64 = job_input.get("init_image")

            if not init_image_b64:
                return {"status": "error", "error": "init_image required for img2img workflow"}

            # Validate strength
            if not 0.0 <= strength <= 1.0:
                return {"status": "error", "error": "strength must be between 0.0 and 1.0"}

            init_image = base64_to_image(init_image_b64)

            result = run_img2img(
                prompt, negative_prompt, init_image, strength,
                width, height, num_inference_steps, guidance_scale,
                num_images, generator
            )

        elif workflow == "multi_reference":
            reference_images_b64 = job_input.get("reference_images", [])
            reference_weights = job_input.get("reference_weights")

            if not reference_images_b64:
                return {"status": "error", "error": "reference_images required for multi_reference workflow"}

            reference_images = [base64_to_image(img_b64) for img_b64 in reference_images_b64]

            result = run_multi_reference(
                prompt, negative_prompt, reference_images, reference_weights,
                width, height, num_inference_steps, guidance_scale,
                num_images, generator
            )

        # Convert images to base64
        images_b64 = []
        for img in result.images:
            img_b64 = image_to_base64(img, format=output_format)
            images_b64.append(img_b64)

        inference_time = time.time() - start_time

        return {
            "status": "success",
            "workflow": workflow,
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
                "model": MODEL_NAME,
                "workflow": workflow
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


def run_txt2img(prompt, negative_prompt, width, height, steps, guidance, num_images, generator):
    """Text-to-Image workflow"""
    print(f"Running txt2img: {prompt[:50]}...")

    with torch.inference_mode():
        result = txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            generator=generator
        )
    return result


def run_img2img(prompt, negative_prompt, init_image, strength, width, height, steps, guidance, num_images, generator):
    """Image-to-Image workflow"""
    print(f"Running img2img with strength={strength}: {prompt[:50]}...")

    # Resize init image to target dimensions
    init_image = init_image.resize((width, height), Image.LANCZOS)

    with torch.inference_mode():
        result = img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            generator=generator
        )
    return result


def run_multi_reference(prompt, negative_prompt, reference_images, weights, width, height, steps, guidance, num_images, generator):
    """
    Multi-reference image workflow
    Combines multiple reference images with weighted influence
    """
    print(f"Running multi_reference with {len(reference_images)} references...")

    # If no weights provided, use equal weights
    if weights is None:
        weights = [1.0 / len(reference_images)] * len(reference_images)

    if len(weights) != len(reference_images):
        raise ValueError("Number of weights must match number of reference images")

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Resize all reference images
    resized_refs = [img.resize((width, height), Image.LANCZOS) for img in reference_images]

    # Blend reference images based on weights
    blended = Image.new('RGB', (width, height))

    for ref_img, weight in zip(resized_refs, weights):
        if blended.getbbox() is None:  # First image
            blended = Image.blend(Image.new('RGB', (width, height)), ref_img, weight)
        else:
            blended = Image.blend(blended, ref_img, weight / (1.0 + weight))

    # Use blended image as init_image with low strength for guidance
    with torch.inference_mode():
        result = img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=blended,
            strength=0.4,  # Lower strength to preserve reference influence
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            generator=generator
        )
    return result


if __name__ == "__main__":
    # Start the serverless function
    runpod.serverless.start({"handler": handler})
