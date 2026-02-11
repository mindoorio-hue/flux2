# Troubleshooting Guide

## Common Errors and Solutions

### ❌ Error: `module 'torch' has no attribute 'xpu'`

**Full Error:**
```
AttributeError: module 'torch' has no attribute 'xpu'
```

**Cause:** Version mismatch between PyTorch and diffusers library.

**Solution:** ✅ FIXED in commit `10e5ab5`
- Pinned `torch==2.1.0` (matches base image)
- Pinned `diffusers==0.29.2` (compatible version)
- All ML libraries now have fixed versions

**Status:** Resolved - rebuild your Docker image

---

### ❌ Error: Docker build fails with base image not found

**Full Error:**
```
failed to resolve source metadata for docker.io/runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04: not found
```

**Cause:** Old/incorrect base image in Dockerfile

**Solution:** ✅ FIXED
- Use `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`
- Clear GitHub Actions cache
- Rebuild

**Status:** Resolved

---

### ❌ Error: Out of Memory (OOM) during inference

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce image dimensions:**
   ```json
   {
     "width": 768,
     "height": 768
   }
   ```

2. **Generate fewer images:**
   ```json
   {
     "num_images": 1
   }
   ```

3. **Use FLUX.1-schnell instead of dev:**
   ```json
   {
     "MODEL_NAME": "black-forest-labs/FLUX.1-schnell"
   }
   ```

4. **Enable CPU offload in handler.py:**
   ```python
   pipe.enable_sequential_cpu_offload()
   ```

---

### ❌ Error: HuggingFace authentication failed

**Error:**
```
401 Unauthorized: Access to model is restricted
```

**Solution:**

1. **Get HuggingFace token:**
   - Go to: https://huggingface.co/settings/tokens
   - Create new token with read access
   - Accept FLUX.1-dev license

2. **Set environment variable:**
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

3. **In RunPod:**
   - Add `HF_TOKEN` in environment variables
   - Value: your token

---

### ❌ Error: Model download is slow/timing out

**Error:**
```
ConnectionError: Timeout while downloading model
```

**Solutions:**

1. **Increase timeout:**
   - RunPod: Set worker timeout to 600+ seconds
   - Give more time for first cold start

2. **Pre-download model in Dockerfile:**
   ```dockerfile
   ENV HF_HOME=/app/models
   RUN python -c "from diffusers import FluxPipeline; \
       FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')"
   ```

3. **Use volume mount:**
   - Mount `/app/models` volume
   - Persist models across restarts

---

### ❌ Error: Import errors for diffusers components

**Error:**
```
ImportError: cannot import name 'FluxImg2ImgPipeline' from 'diffusers'
```

**Cause:** Diffusers version doesn't have Img2Img pipeline

**Solution:**

1. **Update to compatible version:**
   ```bash
   pip install diffusers==0.29.2
   ```

2. **Or remove img2img temporarily:**
   ```python
   # Comment out if not needed:
   # from diffusers import FluxImg2ImgPipeline
   ```

---

### ❌ Error: Base64 decode error

**Error:**
```
binascii.Error: Invalid base64-encoded string
```

**Solutions:**

1. **Check image encoding:**
   ```python
   # Correct format:
   "init_image": "base64_string_without_data_uri_prefix"

   # NOT this:
   "init_image": "data:image/png;base64,..."
   ```

2. **Remove data URI prefix if present:**
   ```python
   if init_image.startswith('data:'):
       init_image = init_image.split(',')[1]
   ```

---

### ❌ Error: Dimensions not multiple of 8

**Error:**
```
Width and height must be multiples of 8
```

**Solution:**
```json
{
  "width": 1024,   // ✅ Valid (multiple of 8)
  "height": 768,   // ✅ Valid
  // NOT:
  "width": 1023,   // ❌ Invalid
  "height": 765    // ❌ Invalid
}
```

---

### ❌ Error: GitHub Actions build timeout

**Error:**
```
Error: The operation was canceled
```

**Solutions:**

1. **Increase workflow timeout:**
   ```yaml
   jobs:
     build:
       timeout-minutes: 60  # Increase from default
   ```

2. **Remove model pre-download:**
   - Don't download models during build
   - Download on first run instead

3. **Use smaller base image:**
   - Use `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
   - Instead of `-devel` variant

---

### ❌ Error: Tests failing in GitHub Actions

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** ✅ FIXED in commit `1a606f6`
- Tests no longer require ML dependencies
- Only syntax and structure checks

---

## Debugging Tips

### 1. Check Docker Logs

```bash
# Local
docker logs <container-id>

# RunPod
# View logs in RunPod dashboard
```

### 2. Test Locally First

```bash
# Build and run locally
docker build -t flux2-test .
docker run --gpus all -e HF_TOKEN=xxx flux2-test

# Test handler directly
python handler.py
```

### 3. Verify Environment Variables

```python
import os
print(f"HF_TOKEN: {os.getenv('HF_TOKEN', 'NOT SET')}")
print(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'NOT SET')}")
```

### 4. Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 5. Monitor VRAM Usage

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

## Performance Issues

### Slow Inference (>60 seconds)

**Check:**
1. Using correct GPU (A40/A100, not CPU)
2. Model loaded globally (not per request)
3. Using appropriate number of steps:
   - FLUX.1-schnell: 1-4 steps
   - FLUX.1-dev: 28-50 steps

### High VRAM Usage

**Solutions:**
1. Enable attention slicing (already enabled)
2. Use mixed precision (bf16)
3. Reduce batch size (num_images)
4. Use smaller image dimensions

---

## Getting Help

### Check Status
- **GitHub Actions**: https://github.com/mindoorio-hue/flux2/actions
- **Container Logs**: RunPod dashboard
- **Build Logs**: GitHub Actions workflow logs

### Useful Commands

```bash
# Check Docker image
docker inspect ghcr.io/mindoorio-hue/flux2:main

# Test endpoint locally
curl -X POST http://localhost:8000/test

# View logs
docker logs -f <container-id>

# Check environment
docker exec <container-id> env
```

---

## Version Information

### Current Versions (✅ Working)
```
torch==2.1.0
torchvision==0.16.0
diffusers==0.29.2
transformers==4.44.0
accelerate==0.33.0
Base Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

### Tested GPU Configurations
- ✅ NVIDIA A40 (48GB VRAM)
- ✅ NVIDIA A100 (40GB/80GB VRAM)
- ✅ NVIDIA RTX 4090 (24GB VRAM)
- ⚠️ NVIDIA RTX 3090 (24GB VRAM) - works with smaller dimensions

---

## Quick Fixes Reference

| Issue | Quick Fix |
|-------|-----------|
| `xpu` error | Use pinned requirements.txt (commit 10e5ab5) |
| Base image not found | Clear cache, rebuild |
| OOM error | Reduce dimensions, use schnell model |
| Import error | Check diffusers version |
| Auth error | Set HF_TOKEN correctly |
| Slow download | Increase timeout, pre-download model |
| Test failures | Use updated test workflow |

---

## Still Having Issues?

1. **Check latest commit**: Ensure you're using latest code
2. **Clear caches**: GitHub Actions → Caches → Delete all
3. **Rebuild fresh**: `docker build --no-cache`
4. **Check logs**: Full error traceback helps
5. **Open issue**: https://github.com/mindoorio-hue/flux2/issues
