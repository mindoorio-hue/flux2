# Verify and Build Guide

## Where are you building from?

### Scenario 1: Building Locally (Your Computer)

1. **Verify you're in the right directory:**
   ```bash
   cd /c/Users/mindoor/Documents/flux2-endpoint
   pwd  # Should show: /c/Users/mindoor/Documents/flux2-endpoint
   ```

2. **Verify Dockerfile is updated:**
   ```bash
   head -5 Dockerfile
   ```

   Should show:
   ```
   # Use official PyTorch CUDA image
   # Alternative RunPod-compatible base images:
   # - pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   # - nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   ```

3. **Start Docker Desktop** (IMPORTANT!)
   - Windows: Open Docker Desktop from Start Menu
   - Wait until it says "Docker Desktop is running"

4. **Build:**
   ```bash
   docker build -t flux2-endpoint .
   ```

---

### Scenario 2: Building from GitHub (RunPod, Actions, etc.)

If you're building **directly from GitHub URL** (common with RunPod):

**Problem**: GitHub may have cached the old Dockerfile

**Solutions**:

#### Option A: Force GitHub to refresh
```bash
# Clear GitHub cache by adding a commit
cd /c/Users/mindoor/Documents/flux2-endpoint
git commit --allow-empty -m "Trigger rebuild"
git push origin main
```

#### Option B: Clone fresh and verify
```bash
# Clone to a new location
cd /tmp
git clone https://github.com/mindoorio-hue/flux2.git
cd flux2

# Check Dockerfile
head -5 Dockerfile

# Should show pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

#### Option C: Use specific commit/branch
When specifying the repo URL in RunPod:
```
https://github.com/mindoorio-hue/flux2.git#main
# or with specific commit:
https://github.com/mindoorio-hue/flux2.git#42df8b3
```

---

### Scenario 3: Building from Different Directory

If you cloned the repo elsewhere:

```bash
# Find all clones
cd ~
find . -name "flux2" -type d 2>/dev/null

# Go to the correct one
cd /path/to/correct/flux2

# Pull latest
git pull origin main

# Verify Dockerfile
head -5 Dockerfile

# Build
docker build -t flux2-endpoint .
```

---

## Quick Fix Commands

### If Docker Desktop isn't running:
```bash
# Windows: Just open Docker Desktop from Start Menu
# It will start automatically
```

### If you're getting cached old version:
```bash
cd /c/Users/mindoor/Documents/flux2-endpoint

# Pull latest from GitHub (in case something went wrong)
git fetch origin
git reset --hard origin/main

# Verify
head -5 Dockerfile

# Build without cache
docker build --no-cache -t flux2-endpoint .
```

### If you want to be ABSOLUTELY sure:
```bash
# Delete and re-clone
cd /c/Users/mindoor/Documents
rm -rf flux2-endpoint
git clone https://github.com/mindoorio-hue/flux2.git flux2-endpoint
cd flux2-endpoint

# Verify
head -5 Dockerfile

# Build
docker build -t flux2-endpoint .
```

---

## Error Messages and What They Mean

### "runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04: not found"
❌ You're using the OLD Dockerfile

**Solution**:
1. Pull latest: `git pull origin main`
2. Verify: `head -5 Dockerfile` should show `pytorch/pytorch:2.1.0`

### "error during connect: ... dockerDesktopLinuxEngine"
❌ Docker Desktop is not running

**Solution**: Start Docker Desktop and wait for it to fully start

### "no such file or directory"
❌ You're in the wrong directory

**Solution**: `cd /c/Users/mindoor/Documents/flux2-endpoint`

---

## Verify GitHub Has Latest

Check on GitHub directly:
1. Go to https://github.com/mindoorio-hue/flux2
2. Click on `Dockerfile`
3. First lines should be:
   ```
   # Use official PyTorch CUDA image
   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   ```

If it still shows `runpod/pytorch:2.1.1...`, then:
```bash
cd /c/Users/mindoor/Documents/flux2-endpoint
git push -f origin main  # Force push
```

---

## Test Without Building

Just verify Docker works:
```bash
# Test Docker is working
docker run hello-world

# Test PyTorch image exists
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

---

## Alternative: Use Pre-built Base

If all else fails, create a minimal Dockerfile:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=black-forest-labs/FLUX.1-dev

CMD ["python", "-u", "handler.py"]
```

Save as `Dockerfile.simple` and build with:
```bash
docker build -f Dockerfile.simple -t flux2-endpoint .
```

---

## Need Help?

Tell me:
1. **Where are you building?** (Local computer, RunPod, GitHub Actions)
2. **What directory are you in?** (`pwd` output)
3. **What does this show?** `head -5 Dockerfile`
4. **Is Docker Desktop running?**

This will help me diagnose the exact issue!
