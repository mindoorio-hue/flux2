# Flux2 Image Generation Endpoint

A production-ready RunPod serverless endpoint for Flux2 image generation with full feature support.

## Features

- ✅ Support for both FLUX.1-dev and FLUX.1-schnell models
- ✅ Customizable image dimensions (any multiple of 8)
- ✅ Adjustable inference steps and guidance scale
- ✅ Multiple image generation in single request
- ✅ Seed control for reproducible results
- ✅ Negative prompts support
- ✅ Multiple output formats (PNG, JPEG, WEBP)
- ✅ Memory-optimized for GPU inference
- ✅ Base64 encoded output for easy API consumption
- ✅ Comprehensive error handling and logging

## Models

### FLUX.1-dev
- High-quality image generation
- Requires 28-50 steps for best results
- Better prompt adherence
- Requires HuggingFace token (gated model)
- Recommended guidance scale: 7.5

### FLUX.1-schnell
- Fast image generation (1-4 steps)
- Good quality for quick results
- No HuggingFace token required
- Recommended guidance scale: 3.5

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your HuggingFace token

# Test locally
python local_test.py
```

### 2. Docker Build & Test

```bash
# Build image
docker build -t flux2-endpoint .

# Test with GPU
docker run --gpus all -it --rm \
  -e HF_TOKEN=your_token_here \
  flux2-endpoint

# For FLUX.1-schnell (faster, no token needed)
docker run --gpus all -it --rm \
  -e MODEL_NAME=black-forest-labs/FLUX.1-schnell \
  flux2-endpoint
```

### 3. Deploy to RunPod

```bash
# Push to container registry
docker tag flux2-endpoint your-registry/flux2-endpoint:latest
docker push your-registry/flux2-endpoint:latest

# Configure in RunPod dashboard:
# - Container Image: your-registry/flux2-endpoint:latest
# - GPU: A40 or better (24GB+ VRAM recommended)
# - Environment Variables: HF_TOKEN, MODEL_NAME
# - Timeout: 600 seconds
```

## API Usage

### Request Format

```json
{
  "input": {
    "prompt": "A majestic lion on a cliff at sunset",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_images": 1,
    "seed": 42,
    "output_format": "PNG"
  }
}
```

### Response Format

```json
{
  "status": "success",
  "images": ["base64_encoded_image_data..."],
  "metadata": {
    "prompt": "...",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "inference_time": 12.34,
    "model": "black-forest-labs/FLUX.1-dev"
  }
}
```

### cURL Example

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains",
      "width": 1024,
      "height": 768,
      "num_inference_steps": 50,
      "guidance_scale": 7.5
    }
  }'
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of image to generate |
| `negative_prompt` | string | "" | What to avoid in the image |
| `width` | int | 1024 | Image width (must be multiple of 8) |
| `height` | int | 1024 | Image height (must be multiple of 8) |
| `num_inference_steps` | int | 50 (dev) / 4 (schnell) | Number of denoising steps |
| `guidance_scale` | float | 7.5 | How closely to follow the prompt |
| `num_images` | int | 1 | Number of images to generate (1-4) |
| `seed` | int | random | Random seed for reproducibility |
| `output_format` | string | "PNG" | Output format: PNG, JPEG, or WEBP |
| `output_quality` | int | 95 | JPEG quality (1-100) |

## Performance Optimization

### GPU Requirements
- **Minimum**: RTX 3090 / A40 (24GB VRAM)
- **Recommended**: A100 (40GB+ VRAM) for best performance
- **Budget**: Can run on 16GB VRAM with reduced batch sizes

### Memory Optimizations Enabled
- Attention slicing
- Mixed precision (bfloat16)
- Optional: Sequential CPU offload (for lower VRAM)

### Inference Times (approximate)
- **FLUX.1-schnell**: 2-5 seconds (4 steps, 1024x1024, A40)
- **FLUX.1-dev**: 15-30 seconds (50 steps, 1024x1024, A40)

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=handler --cov=src tests/

# Test locally with sample input
python local_test.py
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce image dimensions (e.g., 768x768)
- Reduce `num_images` to 1
- Enable CPU offload in handler.py
- Use FLUX.1-schnell instead of dev

### Slow Generation
- Use FLUX.1-schnell for faster results
- Reduce `num_inference_steps`
- Use smaller image dimensions

### Quality Issues
- Increase `num_inference_steps` (dev: 50-100, schnell: 4-8)
- Adjust `guidance_scale` (higher = more prompt adherence)
- Use negative prompts to avoid unwanted elements
- Try different seeds

## Project Structure

```
flux2-endpoint/
├── handler.py              # Main serverless handler
├── local_test.py          # Local testing script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── test_input.json       # Sample input (dev model)
├── test_schnell.json     # Sample input (schnell model)
├── src/                  # Additional modules
│   └── utils/           # Utility functions
├── tests/               # Unit tests
│   └── test_handler.py
└── builder/             # Build scripts
```

## License

This endpoint implementation is provided as-is. Flux models are subject to their respective licenses from Black Forest Labs.
