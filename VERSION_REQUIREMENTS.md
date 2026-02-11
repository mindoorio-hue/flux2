# Flux2 Version Requirements

## Why PyTorch 2.3 is Required

Flux is a brand new model from Black Forest Labs (2024) and has specific version requirements.

### The Version Chain

```
Flux Model
  └─ requires FluxPipeline
      └─ available in diffusers >= 0.30.0
          └─ requires PyTorch >= 2.2.0 (for 'xpu' support)
              └─ best with PyTorch 2.3.0 (latest stable)
```

---

## Failed Attempts & Why

### ❌ Attempt 1: PyTorch 2.1.0 + diffusers 0.30.0
**Error:** `module 'torch' has no attribute 'xpu'`
**Why:** diffusers 0.30+ uses `torch.xpu` which doesn't exist in PyTorch 2.1.0

### ❌ Attempt 2: PyTorch 2.1.0 + diffusers 0.29.2
**Error:** `cannot import name 'FluxPipeline' from 'diffusers'`
**Why:** FluxPipeline was added in diffusers 0.30.0

### ✅ Solution: PyTorch 2.3.0 + diffusers 0.30.0
**Status:** Working!
**Why:** PyTorch 2.3 supports all features needed by diffusers 0.30+

---

## Current Working Configuration

### Base Image
```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
```

### Python Packages
```
torch >= 2.3.0
torchvision >= 0.18.0
diffusers >= 0.30.0
transformers >= 4.44.0
accelerate >= 0.33.0
```

### CUDA Version
- **CUDA 12.1** (comes with PyTorch 2.3 base image)
- Compatible with: A40, A100, RTX 4090, RTX 3090, etc.

---

## Version History

| Date | PyTorch | diffusers | Status | Issue |
|------|---------|-----------|--------|-------|
| Initial | 2.1.0 | 0.30.0 | ❌ Failed | `xpu` attribute error |
| Attempt 2 | 2.1.0 | 0.29.2 | ❌ Failed | FluxPipeline not found |
| **Current** | **2.3.0** | **0.30.0** | **✅ Working** | All features available |

---

## Why This Matters

### FluxPipeline Features
FluxPipeline is the main interface for:
- Text-to-Image generation
- Base for FluxImg2ImgPipeline
- Required for all Flux workflows

### Newer PyTorch Benefits
- Better GPU memory management
- Faster inference with torch.compile()
- Support for latest CUDA features
- Required for modern diffusion libraries

---

## Alternative Approaches (Not Recommended)

### Option 1: Use Stable Diffusion Instead
```python
from diffusers import StableDiffusionPipeline
# Works with older versions, but not Flux
```

### Option 2: Manual Model Loading
```python
# Load Flux transformer manually
# Much more complex, no benefits
```

### Option 3: Older Base Image
```dockerfile
# Use CUDA 11.8 with PyTorch 2.3
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel
# May work, but CUDA 12.1 is better
```

**Recommendation:** Use the current config (PyTorch 2.3 + CUDA 12.1)

---

## GPU Compatibility

### CUDA 12.1 Compatible GPUs
✅ NVIDIA A100 (all variants)
✅ NVIDIA A40
✅ NVIDIA RTX 4090
✅ NVIDIA RTX 4080
✅ NVIDIA RTX 3090
✅ NVIDIA RTX 3080
✅ Most modern data center GPUs

### Minimum Requirements
- Compute Capability: 7.0+
- VRAM: 24GB+ recommended (16GB minimum with optimizations)

---

## Testing Version Compatibility

### Check PyTorch Version
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Check diffusers Version
```python
import diffusers
print(f"diffusers: {diffusers.__version__}")

# Check if FluxPipeline exists
try:
    from diffusers import FluxPipeline
    print("✓ FluxPipeline available")
except ImportError:
    print("✗ FluxPipeline not available")
```

### Expected Output
```
PyTorch: 2.3.0+cu121
CUDA: 12.1
CUDA available: True
diffusers: 0.30.x
✓ FluxPipeline available
```

---

## Migration Guide

### From PyTorch 2.1 → 2.3

**Changes:**
1. CUDA 11.8 → 12.1
2. Updated CuDNN
3. New torch features

**Impact:**
- Minimal code changes
- Better performance
- More memory efficient

**Breaking Changes:**
- None for standard usage
- Some deprecated APIs removed

### From diffusers 0.29 → 0.30

**New Features:**
- FluxPipeline
- FluxImg2ImgPipeline
- Improved memory management

**Breaking Changes:**
- Requires PyTorch 2.2+
- Some scheduler API changes (minor)

---

## Future Considerations

### When to Update Again

Update PyTorch when:
- New CUDA version required
- Significant performance improvements
- Security updates

Update diffusers when:
- New Flux features added
- Bug fixes for Flux
- Performance optimizations

### Version Pinning Strategy

**Current approach:** Minimum versions (>=)
```
torch>=2.3.0
diffusers>=0.30.0
```

**For production:** Pin exact versions
```
torch==2.3.1
diffusers==0.30.2
```

---

## Troubleshooting Version Issues

### If imports fail
```bash
# Check installed versions
pip list | grep -E "torch|diffusers"

# Expected:
# torch         2.3.x
# diffusers     0.30.x
```

### If CUDA not working
```bash
# Check CUDA in container
nvidia-smi

# Expected: CUDA 12.1+
```

### If building fails
```bash
# Clear all caches
docker system prune -a

# Build without cache
docker build --no-cache -t flux2-test .
```

---

## References

- **PyTorch 2.3 Release Notes**: https://pytorch.org/blog/pytorch2-3/
- **diffusers 0.30 Release**: https://github.com/huggingface/diffusers/releases
- **Flux Model**: https://huggingface.co/black-forest-labs
- **CUDA Compatibility**: https://docs.nvidia.com/cuda/

---

## Summary

✅ **Working Config:**
- Base: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`
- PyTorch: 2.3.0+
- diffusers: 0.30.0+
- CUDA: 12.1

✅ **Why:**
- Flux requires FluxPipeline (diffusers 0.30+)
- diffusers 0.30+ requires PyTorch 2.2+
- PyTorch 2.3 is latest stable with CUDA 12.1

✅ **Result:**
- All 3 workflows working
- Best performance
- Future-proof
