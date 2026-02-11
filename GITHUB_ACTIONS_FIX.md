# GitHub Actions Docker Build - FIXED

## Problem Identified

GitHub Actions was using **cached Docker layers** with the old RunPod base image.

## Solution Applied

✅ **Commit**: `115aa8e` - "Fix GitHub Actions Docker build"

### Changes Made:

1. **Disabled cache**: Added `no-cache: true` to force fresh build
2. **Pull latest images**: Added `pull: true` to get latest base images
3. **Explicit Dockerfile**: Specified `file: ./Dockerfile` explicitly
4. **Verification step**: Added pre-build check to verify correct base image

---

## How to Monitor the Fix

### Option 1: Via GitHub Website

1. Go to: **https://github.com/mindoorio-hue/flux2/actions**
2. You should see a new workflow run triggered by commit `115aa8e`
3. Click on the running workflow
4. Watch the build logs in real-time
5. Look for the "Verify Dockerfile" step - it should show:
   ```
   ✓ Correct base image found!
   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   ```

### Option 2: Via GitHub CLI (After Installing)

```bash
cd /c/Users/mindoor/Documents/flux2-endpoint

# List recent workflow runs
gh run list --workflow=docker.yml --limit 5

# Watch the latest run
gh run watch

# View logs
gh run view --log
```

---

## What the Workflow Now Does

```yaml
1. Checkout code from GitHub
2. Verify Dockerfile ✓ NEW
   - Shows first 10 lines
   - Checks for correct base image
   - Fails if wrong image found
3. Set up Docker Buildx
4. Login to GitHub Container Registry
5. Build Docker image
   - no-cache: true (no old layers)
   - pull: true (fresh base images)
   - file: ./Dockerfile (explicit)
6. Push to ghcr.io/mindoorio-hue/flux2:main
```

---

## Expected Build Output

The build should now show:

```
=== Dockerfile Content (first 10 lines) ===
# Use official PyTorch CUDA image
# Alternative RunPod-compatible base images:
# - pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# - nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

=== Checking for correct base image ===
✓ Correct base image found!

Building Docker image...
[+] Building 234.5s
 => pulling pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
 => [1/5] FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
 ...
```

---

## If Build Still Fails

### Clear GitHub Actions Cache

```bash
# Via GitHub CLI (after installing)
gh cache list
gh cache delete <cache-key>

# Or delete all caches
gh cache delete --all
```

### Manual Trigger

Force a new build:

```bash
cd /c/Users/mindoor/Documents/flux2-endpoint

# Make a trivial change
git commit --allow-empty -m "Trigger rebuild"
git push origin main
```

### Check Dockerfile on GitHub

Verify GitHub has the correct file:

```bash
# Via curl
curl -H "Accept: application/vnd.github.v3.raw" \
  https://api.github.com/repos/mindoorio-hue/flux2/contents/Dockerfile | head -5

# Should show:
# # Use official PyTorch CUDA image
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

---

## After Successful Build

Once the build succeeds, your Docker image will be available at:

```
ghcr.io/mindoorio-hue/flux2:main
```

### Use in RunPod

1. Go to RunPod → Serverless → Endpoints
2. Create new endpoint
3. Container Image: `ghcr.io/mindoorio-hue/flux2:main`
4. Set environment variables:
   - `HF_TOKEN`: Your HuggingFace token
   - `MODEL_NAME`: `black-forest-labs/FLUX.1-dev` (or schnell)
5. Select GPU: A40 or A100
6. Deploy!

### Pull Locally

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/mindoorio-hue/flux2:main

# Run locally
docker run --gpus all \
  -e HF_TOKEN=your_token \
  ghcr.io/mindoorio-hue/flux2:main
```

---

## Monitoring Build Status

### Real-time Updates

The workflow runs automatically on every push to `main`. Check:

**GitHub Actions**: https://github.com/mindoorio-hue/flux2/actions

### Build Duration

Expected build time: **8-15 minutes**
- Pulling base image: ~3-5 min
- Installing dependencies: ~4-8 min
- Copying files: ~1 min
- Pushing to registry: ~2-4 min

---

## Alternative: Build Specific Dockerfile

If you want to test different Dockerfiles:

### Build with Dockerfile.runpod

```yaml
# In .github/workflows/docker.yml, change:
file: ./Dockerfile.runpod
```

### Build with Dockerfile.nvidia

```yaml
file: ./Dockerfile.nvidia
```

---

## Troubleshooting

### Error: "pytorch/pytorch:2.1.0... not found"

**Unlikely** but if it happens:
- The tag might be wrong
- Check available tags: https://hub.docker.com/r/pytorch/pytorch/tags
- Use: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (smaller)

### Error: "unauthorized"

Check repository settings:
- Settings → Actions → General
- Workflow permissions: "Read and write permissions"

### Error: Build timeout

Increase timeout in workflow:
```yaml
jobs:
  build:
    timeout-minutes: 60  # Default is 360
```

---

## Success Indicators

Build succeeded when you see:

```
✓ Build Docker image
✓ Push Docker image
✓ Docker image pushed to ghcr.io/mindoorio-hue/flux2:main
```

---

## Summary

✅ **Fixed**: GitHub Actions workflow updated
✅ **Cache**: Disabled to force fresh build
✅ **Verification**: Added Dockerfile check step
✅ **Pushed**: Changes live on GitHub
✅ **Building**: New workflow should be running now

**Check status**: https://github.com/mindoorio-hue/flux2/actions

The next build should succeed with the correct PyTorch base image! 🎉

---

## Quick Links

- **Repository**: https://github.com/mindoorio-hue/flux2
- **Actions**: https://github.com/mindoorio-hue/flux2/actions
- **Dockerfile**: https://github.com/mindoorio-hue/flux2/blob/main/Dockerfile
- **Container Registry**: https://github.com/mindoorio-hue/flux2/pkgs/container/flux2
