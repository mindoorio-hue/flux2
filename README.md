# FLUX.2 Image Generation Endpoint

A production-ready RunPod serverless endpoint for FLUX.2 image generation with comprehensive workflow support.

## Features

### 🎨 **Three Complete Workflows**
- ✅ **Text-to-Image**: Generate images from text prompts
- ✅ **Image-to-Image**: Transform existing images with prompts and strength control
- ✅ **Multi-Reference**: Combine multiple reference images with weighted influence

### 🚀 **Model Support**
- ✅ **FLUX.2-dev**: High-quality generation (28-50 steps, gated model)
- ✅ **FLUX.2-schnell**: Fast generation (1-4 steps, open access)
- ✅ **FLUX.1**: Backward compatible with dual-encoder architecture
- ✅ **4-bit Quantization**: NF4 quantization via bitsandbytes for 44GB GPU compatibility
- ✅ **Model CPU Offload**: Component-level offloading for optimal memory management

### 🎛️ **Generation Control**
- ✅ Customizable image dimensions (any multiple of 8, up to 2048x2048)
- ✅ Adjustable inference steps (1-100)
- ✅ Guidance scale control (0.0-20.0)
- ✅ Multiple images per request (1-4)
- ✅ Seed control for reproducible results
- ✅ Negative prompts support (FLUX.1 only)
- ✅ Strength parameter for image-to-image (0.0-1.0)
- ✅ Reference image weights for multi-reference workflow

### 💾 **Output & Performance**
- ✅ Multiple output formats (PNG, JPEG, WEBP)
- ✅ Base64 encoded output for easy API consumption
- ✅ Inline response delivery (no S3 uploads)
- ✅ Memory-optimized inference (attention slicing, VAE tiling)
- ✅ Expandable GPU memory segments for large models
- ✅ bfloat16 precision for speed and efficiency

### 🛡️ **Reliability**
- ✅ Comprehensive input validation
- ✅ Detailed error handling and logging
- ✅ Automatic workflow detection
- ✅ Metadata tracking (prompt, seed, inference time, model info)
- ✅ RunPod async polling with timeout protection

## Models

### FLUX.2-dev
- High-quality image generation
- Requires 28-50 steps for best results
- Better prompt adherence
- Requires HuggingFace token (gated model)
- Recommended guidance scale: 7.5

### FLUX.2-schnell
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

# For FLUX.2-schnell (faster, no token needed)
docker run --gpus all -it --rm \
  -e MODEL_NAME=black-forest-labs/FLUX.2-schnell \
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

### Text-to-Image Workflow

Generate images from text prompts.

```json
{
  "input": {
    "prompt": "A majestic lion on a cliff at sunset, cinematic lighting, highly detailed",
    "negative_prompt": "blurry, low quality, distorted",
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

### Image-to-Image Workflow

Transform existing images with prompts and strength control.

```json
{
  "input": {
    "prompt": "Transform this into a watercolor painting style",
    "init_image": "base64_encoded_image_data...",
    "strength": 0.8,
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_images": 1,
    "seed": 42
  }
}
```

**Parameters:**
- `init_image` (required): Base64 encoded source image
- `strength` (optional, default: 0.8): Transformation strength (0.0-1.0)
  - 0.0 = keep original image
  - 1.0 = maximum transformation

### Multi-Reference Workflow

Combine multiple reference images with weighted influence.

```json
{
  "input": {
    "prompt": "A stylized portrait combining elements from the references",
    "reference_images": [
      "base64_encoded_image_1...",
      "base64_encoded_image_2...",
      "base64_encoded_image_3..."
    ],
    "reference_weights": [0.5, 0.3, 0.2],
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_images": 1,
    "seed": 42
  }
}
```

**Parameters:**
- `reference_images` (required): Array of base64 encoded images
- `reference_weights` (optional): Weight for each reference (must sum to 1.0)
  - If not provided, equal weights are used
  - FLUX.2: Native multi-image support
  - FLUX.1: Blends images then applies img2img

### Response Format

```json
{
  "status": "success",
  "workflow": "txt2img",
  "images": ["base64_encoded_image_data..."],
  "metadata": {
    "prompt": "...",
    "negative_prompt": "...",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42,
    "num_images": 1,
    "inference_time": 45.67,
    "model": "black-forest-labs/FLUX.2-dev",
    "workflow": "txt2img"
  }
}
```

### cURL Examples

#### Text-to-Image
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains, dramatic lighting, 8k uhd",
      "width": 1024,
      "height": 768,
      "num_inference_steps": 50,
      "guidance_scale": 7.5,
      "seed": 12345
    }
  }'
```

#### Image-to-Image
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Transform into anime style, vibrant colors",
      "init_image": "'"$(base64 -w 0 input.jpg)"'",
      "strength": 0.75,
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 40,
      "guidance_scale": 7.0
    }
  }'
```

#### Multi-Reference
```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A character design combining elements from all references",
      "reference_images": [
        "'"$(base64 -w 0 ref1.jpg)"'",
        "'"$(base64 -w 0 ref2.jpg)"'"
      ],
      "reference_weights": [0.6, 0.4],
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 7.5
    }
  }'
```

**Note:** The endpoint uses async polling. After submission:
1. Get the job ID from the response
2. Poll `/status/{job_id}` endpoint until status is "COMPLETED"
3. Extract images from the final response

## Parameters Reference

### Common Parameters (All Workflows)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of image to generate |
| `negative_prompt` | string | "" | What to avoid in the image (FLUX.1 only) |
| `width` | int | 1024 | Image width (must be multiple of 8) |
| `height` | int | 1024 | Image height (must be multiple of 8) |
| `num_inference_steps` | int | 50 (dev) / 4 (schnell) | Number of denoising steps |
| `guidance_scale` | float | 7.5 | How closely to follow the prompt |
| `num_images` | int | 1 | Number of images to generate (1-4) |
| `seed` | int | random | Random seed for reproducibility |
| `output_format` | string | "PNG" | Output format: PNG, JPEG, or WEBP |

### Image-to-Image Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_image` | string | **required** | Base64 encoded initial image |
| `strength` | float | 0.8 | Transformation strength (0.0-1.0) |

### Multi-Reference Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_images` | array | **required** | Array of base64 encoded images |
| `reference_weights` | array | equal weights | Weight for each reference (must sum to 1.0) |

## Performance Optimization

### GPU Requirements
- **FLUX.2 with 4-bit Quantization**: 44GB VRAM minimum
  - NVIDIA A40 (48GB) - Recommended
  - NVIDIA A100 40GB/80GB - Excellent
  - NVIDIA RTX A6000 (48GB) - Good
- **FLUX.1**: 24GB VRAM minimum
  - RTX 3090 / RTX 4090 (24GB)
  - A40 (48GB)

### Memory Optimizations Enabled
- **4-bit NF4 quantization** (bitsandbytes) for FLUX.2
- **Model CPU offload** (component-level, not sequential)
- **Attention slicing** for reduced memory during inference
- **VAE tiling** for large images
- **Mixed precision (bfloat16)** for speed and efficiency
- **Expandable GPU segments** for better memory management

### Inference Times (approximate, A40 48GB)
- **FLUX.2-schnell**: 8-15 seconds (4 steps, 1024x1024)
- **FLUX.2-dev**: 45-90 seconds (50 steps, 1024x1024)
- **Image-to-Image**: Similar to text-to-image
- **Multi-Reference**: +10-20% overhead for multiple images

**Note:** First request after cold start adds 30-60s for model loading.

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
- Use FLUX.2-schnell instead of dev

### Slow Generation
- Use FLUX.2-schnell for faster results
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
├── handler.py                  # Main serverless handler (all 3 workflows)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── README.md                   # This file
├── hub.json                    # RunPod Hub configuration
├── tests.json                  # RunPod Hub test definitions
├── icon.png                    # RunPod Hub icon
├── VERSION_REQUIREMENTS.md     # PyTorch/CUDA version documentation
├── local_test.py              # Local testing script
├── demo.html                  # Web UI for testing (local only)
├── sample.html                # Reference implementation
├── test_input.json            # Sample input (dev model)
├── test_schnell.json          # Sample input (schnell model)
├── src/                       # Additional modules
│   └── utils/                # Utility functions
├── tests/                     # Unit tests
│   └── test_handler.py
└── builder/                   # Build scripts
```

### Key Files
- **handler.py**: Unified handler supporting txt2img, img2img, and multi-reference
- **hub.json**: RunPod template configuration with presets for dev/schnell
- **tests.json**: Validation tests for RunPod Hub publishing
- **demo.html**: Dark mode web UI with all workflows, seed control, request/response display

## Architecture & Workflow Detection

### Automatic Workflow Detection

The handler automatically detects which workflow to use based on input parameters:

```python
def detect_workflow(job_input: dict) -> str:
    has_init_image = job_input.get("init_image") is not None
    has_reference_images = job_input.get("reference_images") is not None

    if has_reference_images:
        return "multi_reference"
    elif has_init_image:
        return "img2img"
    else:
        return "txt2img"
```

### FLUX.1 vs FLUX.2 Architecture

**FLUX.2** (single encoder):
- Uses unified `Flux2Pipeline` for all workflows
- No separate img2img pipeline needed
- Image guidance via `image` parameter
- Native multi-image support (pass list of images)
- No negative prompt support
- Guidance scale adjusted lower for image conditioning (×0.6)

**FLUX.1** (dual encoder):
- Separate pipelines: `FluxPipeline` and `FluxImg2ImgPipeline`
- Has `text_encoder_2` and `tokenizer_2`
- Traditional img2img with `strength` parameter
- Multi-reference via image blending + img2img
- Supports negative prompts

The handler detects which architecture is loaded and adapts behavior automatically:

```python
has_dual_encoders = hasattr(txt2img_pipe, 'text_encoder_2')
```

## Publishing to RunPod Hub

This repository is ready for RunPod Hub publishing:

### Required Files (✅ Included)
- ✅ `hub.json` - Template configuration with presets
- ✅ `tests.json` - Validation tests
- ✅ `handler.py` - Serverless function
- ✅ `Dockerfile` - Container build
- ✅ `README.md` - Documentation
- ✅ `icon.png` - Template icon

### Publishing Steps

1. **Create GitHub Release**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Submit to RunPod Hub**
   - Go to RunPod Hub
   - Click "Submit Template"
   - Provide GitHub repository URL
   - Wait for automated build and tests

3. **Review Process**
   - Status: "Pending" during build/test
   - Automated tests run (defined in tests.json)
   - Manual review by RunPod team
   - Status changes to "Published" when approved

### Template Presets

Two presets are configured in `hub.json`:

1. **FLUX.2-dev (High Quality)**
   - 28-50 steps, best quality
   - Requires HuggingFace token
   - Recommended: A40, A100

2. **FLUX.2-schnell (Fast)**
   - 1-4 steps, fast generation
   - No token required
   - Recommended: A40, A100, RTX A6000

## License

This endpoint implementation is provided as-is. Flux models are subject to their respective licenses from Black Forest Labs.
