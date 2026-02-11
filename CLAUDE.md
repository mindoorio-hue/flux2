# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-ready RunPod serverless endpoint for **Flux2 image generation**. Flux is a state-of-the-art text-to-image model from Black Forest Labs, available in two variants:
- **FLUX.1-dev**: High-quality, slower (28-50 steps), gated model
- **FLUX.1-schnell**: Fast generation (1-4 steps), open access

## Architecture

### Core Components

**handler.py**: Main entry point
- Loads Flux pipeline globally (persists across requests)
- Processes image generation requests
- Handles validation, generation, and base64 encoding
- Returns structured responses with metadata

**Key Design Patterns**:
- Model loaded once at startup (cold start optimization)
- Memory optimizations enabled (attention slicing, bf16)
- Comprehensive input validation
- Seed management for reproducibility
- Base64 encoding for API compatibility

### Model Loading Strategy
```python
# Global scope - loaded once per container lifetime
pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
pipe.enable_attention_slicing()  # Memory optimization
```

This ensures the model persists across requests, avoiding reload overhead.

## Development Commands

### Local Testing
```bash
# Quick test with sample input
python local_test.py

# Test with custom input
python -c "import json; from handler import handler; \
  event = {'input': {'prompt': 'test', 'width': 512, 'height': 512, 'num_inference_steps': 4}}; \
  print(handler(event))"

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run with coverage
pytest --cov=handler --cov=src tests/ -v
```

### Docker Development
```bash
# Build for FLUX.1-dev (default)
docker build -t flux2-endpoint .

# Build for FLUX.1-schnell (faster)
docker build -t flux2-endpoint-schnell \
  --build-arg MODEL_NAME=black-forest-labs/FLUX.1-schnell .

# Test locally (requires GPU)
docker run --gpus all -it --rm \
  -e HF_TOKEN=your_token \
  flux2-endpoint

# Test without GPU (slow, for debugging)
docker run -it --rm \
  -e HF_TOKEN=your_token \
  flux2-endpoint
```

### Deployment
```bash
# Tag and push to registry
docker tag flux2-endpoint your-registry/flux2-endpoint:v1.0.0
docker push your-registry/flux2-endpoint:v1.0.0

# Test deployed endpoint (replace with your endpoint ID and API key)
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## Key Implementation Details

### Input Validation
Critical validations in handler:
- Prompt is required (cannot be empty)
- Width/height must be multiples of 8 (diffusion model requirement)
- num_images must be 1-4 (resource constraint)
- Seed handling: auto-generated if not provided, tracked in metadata

### Model-Specific Defaults
```python
# FLUX.1-dev: 50 steps, guidance 7.5
# FLUX.1-schnell: 4 steps, guidance 3.5
default_steps = 4 if "schnell" in MODEL_NAME.lower() else 50
```

### Memory Management
**Enabled optimizations**:
- `pipe.enable_attention_slicing()`: Reduces VRAM usage
- `torch_dtype=torch.bfloat16`: Half precision (faster, less VRAM)
- `torch.inference_mode()`: Disables gradient computation

**For extreme memory constraints**, uncomment in handler.py:
```python
pipe.enable_sequential_cpu_offload()  # Offload layers to CPU
```

### Output Handling
Images are base64-encoded for API transport:
```python
def image_to_base64(image: Image.Image, format: str = "PNG") -> str
```
Supports PNG (lossless), JPEG (smaller), WEBP (modern, efficient).

## Common Tasks

### Adding LoRA Support
1. Add to requirements.txt: `peft>=0.7.0`
2. In handler.py, before generation:
```python
if lora_path := job_input.get("lora_path"):
    pipe.load_lora_weights(lora_path)
    lora_scale = job_input.get("lora_scale", 0.7)
```

### Adding Image-to-Image
1. Change import: `from diffusers import FluxImg2ImgPipeline`
2. Add input parameter: `init_image` (base64 encoded)
3. Decode and pass to pipeline:
```python
init_img = Image.open(BytesIO(base64.b64decode(init_image)))
result = pipe(prompt=prompt, image=init_img, strength=0.8, ...)
```

### Adding Controlnet
1. Import: `from diffusers import FluxControlNetPipeline`
2. Load controlnet model alongside main pipeline
3. Pass control image in input

### Custom Schedulers
Flux uses custom scheduler, but for experimentation:
```python
from diffusers import EulerDiscreteScheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

## GPU Requirements

| GPU | VRAM | Performance | Notes |
|-----|------|-------------|-------|
| RTX 3090 | 24GB | Good | Minimum for dev model |
| A40 | 48GB | Excellent | Recommended for production |
| A100 | 40-80GB | Excellent | Best performance |
| RTX 4090 | 24GB | Good | Consumer option |

**VRAM Usage** (approximate):
- FLUX.1-dev: ~18GB (1024x1024, bf16)
- FLUX.1-schnell: ~16GB (1024x1024, bf16)

Reduce dimensions or use CPU offload for lower VRAM.

## Error Handling Patterns

All errors return structured format:
```python
{
    "status": "error",
    "error": "User-friendly error message",
    "traceback": "Full stack trace (for debugging)"
}
```

**Common errors to handle**:
- `torch.cuda.OutOfMemoryError`: Reduce batch size, dimensions, or enable offload
- `OSError: No such file`: Model not downloaded (check HF_TOKEN)
- `RuntimeError: Expected tensor`: Input validation failed

## Testing Strategy

**Unit tests** (tests/test_handler.py):
- Mock the pipeline to test logic without model
- Test input validation
- Test error handling

**Integration tests** (local_test.py):
- Full end-to-end with actual model
- Save outputs to verify quality
- Measure inference time

**Run before deployment**:
```bash
pytest tests/  # Fast unit tests
python local_test.py  # Slow integration test
```

## Environment Variables

Set in `.env` locally, RunPod dashboard for deployment:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_NAME` | No | FLUX.1-dev | Model variant to load |
| `HF_TOKEN` | Yes (for dev) | None | HuggingFace access token |
| `HF_HOME` | No | ~/.cache | Model cache directory |
| `CUDA_VISIBLE_DEVICES` | No | 0 | GPU device ID |

## Performance Tips

1. **Cold start optimization**: Models are pre-loaded; first request after container start is fast (~2s overhead)
2. **Batch processing**: Generate multiple images in one request (use `num_images`)
3. **Schnell for speed**: Use FLUX.1-schnell for real-time applications
4. **Dimension tuning**: Smaller dimensions = faster generation (512x512 is 4x faster than 1024x1024)
5. **Step reduction**: Fewer steps = faster (but lower quality). Schnell works well at 1-4 steps.

## Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check RunPod logs**:
- All print() statements appear in RunPod logs
- Check for OOM errors, model loading issues
- Inference time is logged per request

**Local debugging**:
```bash
# Test with minimal resources
python -c "from handler import handler; print(handler({'input': {'prompt': 'test', 'width': 512, 'height': 512, 'num_inference_steps': 1}}))"
```

## Important Notes

- **HuggingFace token**: Required for FLUX.1-dev (gated model). Get from huggingface.co/settings/tokens
- **Model size**: ~24GB download. Ensure sufficient storage and network bandwidth.
- **Cold start**: First request takes 30-60s (model loading). Keep workers warm for production.
- **Costs**: GPU workers are expensive. Use FLUX.1-schnell and batch requests to minimize costs.
- **Rate limits**: RunPod has rate limits. Implement client-side queuing for high load.
