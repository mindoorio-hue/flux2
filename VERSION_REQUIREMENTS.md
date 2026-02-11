# Flux2 Version Requirements

## Why PyTorch 2.5+ is Required

Flux is a brand new model from Black Forest Labs (2024) and has specific version requirements.

### The Version Chain

```
Flux 2 Model (2024+)
  └─ requires FluxPipeline
      └─ available in diffusers >= 0.30.0
          └─ uses torch.nn.RMSNorm + enable_gqa
              └─ RMSNorm added in PyTorch 2.4.0
              └─ enable_gqa added in PyTorch 2.5.0
                  └─ requires PyTorch >= 2.5.0
```

### Critical Requirements

**1. RMSNorm (PyTorch 2.4.0+)**

The Flux transformer architecture uses **RMSNorm** (Root Mean Square Layer Normalization), which was added to PyTorch in version 2.4.0. Earlier versions will fail with:

```
AttributeError: module 'torch.nn' has no attribute 'RMSNorm'
```

**2. Grouped Query Attention (PyTorch 2.5.0+)**

Flux uses **enable_gqa** parameter in `scaled_dot_product_attention()` for efficient attention computation. This was added in PyTorch 2.5.0. Earlier versions will fail with:

```
TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'
```

---

## Failed Attempts & Why

### ❌ Attempt 1: PyTorch 2.1.0 + diffusers 0.30.0
**Error:** `module 'torch' has no attribute 'xpu'`
**Why:** diffusers 0.30+ uses `torch.xpu` which doesn't exist in PyTorch 2.1.0

### ❌ Attempt 2: PyTorch 2.1.0 + diffusers 0.29.2
**Error:** `cannot import name 'FluxPipeline' from 'diffusers'`
**Why:** FluxPipeline was added in diffusers 0.30.0

### ❌ Attempt 3: PyTorch 2.3.0 + diffusers 0.30.0
**Error:** `module 'torch.nn' has no attribute 'RMSNorm'`
**Why:** RMSNorm was added in PyTorch 2.4.0, but Flux transformer uses it

### ❌ Attempt 4: PyTorch 2.4.0 + diffusers 0.30.0
**Error:** `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`
**Why:** enable_gqa parameter was added in PyTorch 2.5.0

### ✅ Solution: PyTorch 2.5.0+ + diffusers 0.30.0
**Status:** Working!
**Why:** PyTorch 2.5+ includes both RMSNorm and enable_gqa required by Flux

---

## Current Working Configuration

### Base Image
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

### Python Packages
```
torch >= 2.5.0
torchvision >= 0.20.0
diffusers >= 0.30.0
transformers >= 4.44.0
accelerate >= 0.33.0
```

### CUDA Version
- **CUDA 12.4** (comes with PyTorch 2.5.1 base image)
- Compatible with: A40, A100, H100, RTX 4090, RTX 3090, etc.

---

## Version History

| Date | PyTorch | diffusers | CUDA | Status | Issue |
|------|---------|-----------|------|--------|-------|
| Initial | 2.1.0 | 0.30.0 | 11.8 | ❌ Failed | `xpu` attribute error |
| Attempt 2 | 2.1.0 | 0.29.2 | 11.8 | ❌ Failed | FluxPipeline not found |
| Attempt 3 | 2.3.0 | 0.30.0 | 12.1 | ❌ Failed | RMSNorm not found |
| Attempt 4 | 2.4.0 | 0.30.0 | 12.1 | ❌ Failed | enable_gqa not found |
| **Current** | **2.5.1** | **0.30.0** | **12.4** | **✅ Working** | All features available |

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
# Use CUDA 11.8 with PyTorch 2.4
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel
# May work, but CUDA 12.1 is better
```

**Recommendation:** Use the current config (PyTorch 2.4 + CUDA 12.1)

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
PyTorch: 2.5.1+cu124
CUDA: 12.4
CUDA available: True
diffusers: 0.30.x
✓ FluxPipeline available
✓ torch.nn.RMSNorm available
✓ enable_gqa parameter available
```

---

## Migration Guide

### From PyTorch 2.4 → 2.5

**Changes:**
1. CUDA 12.1 → 12.4
2. Added enable_gqa parameter to scaled_dot_product_attention (required for Flux)
3. Improved attention performance with GQA (Grouped Query Attention)
4. Better memory efficiency for large models

**Impact:**
- No code changes needed
- GQA support for optimized attention
- Better performance for Flux models
- More memory efficient inference

**Breaking Changes:**
- None for standard usage
- Some deprecated APIs removed

### From PyTorch 2.3 → 2.5

**Changes:**
1. CUDA 12.1 → 12.4
2. CuDNN 8 → 9
3. Added torch.nn.RMSNorm (PyTorch 2.4)
4. Added enable_gqa parameter (PyTorch 2.5)

**Impact:**
- No code changes needed
- Both RMSNorm and GQA now available
- Significantly better performance
- More memory efficient

**Breaking Changes:**
- None for standard usage
- Some deprecated APIs removed

### From PyTorch 2.1 → 2.5

**Changes:**
1. CUDA 11.8 → 12.4
2. CuDNN 8 → 9
3. Major feature additions: RMSNorm, enable_gqa, xpu support

**Impact:**
- Minimal code changes
- Dramatically better performance
- Much more memory efficient
- Full Flux model support

**Breaking Changes:**
- None for standard diffusers usage
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
- Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- PyTorch: 2.5.1+
- diffusers: 0.30.0+
- CUDA: 12.4

✅ **Why:**
- Flux requires FluxPipeline (diffusers 0.30+)
- Flux transformer uses torch.nn.RMSNorm (PyTorch 2.4.0+)
- Flux attention uses enable_gqa parameter (PyTorch 2.5.0+)
- diffusers 0.30+ requires PyTorch 2.5+ for full Flux support
- PyTorch 2.5.1 is latest stable with CUDA 12.4

✅ **Result:**
- All 3 workflows working
- RMSNorm support for Flux transformer
- GQA (Grouped Query Attention) for efficient inference
- Best performance
- Future-proof
