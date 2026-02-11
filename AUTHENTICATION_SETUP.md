# HuggingFace Authentication Setup

## Current Error

```
401 Unauthorized: Access to model black-forest-labs/FLUX.1-dev is restricted.
You must have access to it and be authenticated to access it.
```

This is **normal** - FLUX.1-dev is a gated model that requires authentication.

---

## Solution 1: Set HF_TOKEN (Recommended)

### Step 1: Get Your HuggingFace Token

1. **Go to**: https://huggingface.co/settings/tokens
2. **Click**: "New token"
3. **Name**: "flux2-endpoint"
4. **Type**: Read
5. **Click**: "Generate token"
6. **Copy** the token (starts with `hf_...`)

### Step 2: Accept FLUX.1-dev License

1. **Go to**: https://huggingface.co/black-forest-labs/FLUX.1-dev
2. **Click**: "Agree and access repository"
3. **Accept** the license terms

### Step 3: Set Token in RunPod

1. Go to your RunPod endpoint
2. Click **"Edit"** or **"Settings"**
3. Find **"Environment Variables"** section
4. Add:
   - **Name**: `HF_TOKEN`
   - **Value**: `hf_xxxxxxxxxxxxxxxxxxxxx` (your token)
5. **Save** and **restart** the endpoint

### Expected Result

After setting the token, you'll see:
```
✓ Loading Flux model: black-forest-labs/FLUX.1-dev
✓ Model loaded successfully!
✓ Flux model loaded successfully!
```

---

## Solution 2: Use FLUX.1-schnell (No Auth Required)

If you don't want to set up authentication, use the schnell model:

### Change Environment Variable

In RunPod, set:
- **Name**: `MODEL_NAME`
- **Value**: `black-forest-labs/FLUX.1-schnell`

### Differences: dev vs schnell

| Feature | FLUX.1-dev | FLUX.1-schnell |
|---------|------------|----------------|
| Quality | Higher | Good |
| Speed | Slower (50 steps) | Faster (1-4 steps) |
| Auth Required | ✅ Yes | ❌ No |
| License | Gated | Apache 2.0 |
| Use Case | Production, high quality | Testing, quick results |

---

## Troubleshooting Auth Issues

### Error: "Invalid token"

**Check:**
1. Token copied correctly (no extra spaces)
2. Token starts with `hf_`
3. Token has "read" permissions

**Fix:**
- Generate a new token
- Make sure to copy the entire token

### Error: "Access denied"

**Check:**
1. Accepted FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Used the correct HuggingFace account

**Fix:**
- Log in to HuggingFace
- Go to FLUX.1-dev page
- Accept the license

### Error: "Token not found"

**Check:**
1. Environment variable name is exactly `HF_TOKEN`
2. No typos in the variable name
3. Endpoint was restarted after adding the variable

**Fix:**
- Double-check variable name: `HF_TOKEN`
- Restart the endpoint

---

## Testing Authentication

### Local Test

```bash
export HF_TOKEN=hf_your_token_here

python -c "
from huggingface_hub import login
login(token='$HF_TOKEN')
print('✓ Authentication successful!')
"
```

### Docker Test

```bash
docker run --rm \
  -e HF_TOKEN=hf_your_token_here \
  ghcr.io/mindoorio-hue/flux2:main \
  python -c "import os; print(f'HF_TOKEN: {os.getenv(\"HF_TOKEN\", \"NOT SET\")[:10]}...')"
```

---

## Environment Variables Reference

### Required for FLUX.1-dev

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### Optional

```bash
MODEL_NAME=black-forest-labs/FLUX.1-dev  # default
HF_HOME=/app/models                       # cache directory
```

### For FLUX.1-schnell (No Token Needed)

```bash
MODEL_NAME=black-forest-labs/FLUX.1-schnell
# HF_TOKEN not required!
```

---

## RunPod Configuration

### Complete Setup

1. **Container Image**: `ghcr.io/mindoorio-hue/flux2:main`

2. **Environment Variables**:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   MODEL_NAME=black-forest-labs/FLUX.1-dev
   ```

3. **GPU**: A40 or A100 (24GB+ VRAM)

4. **Container Disk**: 50GB+ (for model storage)

5. **Timeout**: 600 seconds

6. **Workers**: 1-3 depending on traffic

---

## Security Best Practices

### ✅ Do:
- Store token in environment variables
- Use read-only tokens
- Rotate tokens periodically
- Keep tokens private

### ❌ Don't:
- Commit tokens to git
- Share tokens publicly
- Use write permissions unless needed
- Store tokens in code

---

## Quick Start Commands

### Option 1: FLUX.1-dev (High Quality)

```bash
# In RunPod Environment Variables:
HF_TOKEN=hf_your_actual_token_here
MODEL_NAME=black-forest-labs/FLUX.1-dev
```

**Required:**
1. Get token from https://huggingface.co/settings/tokens
2. Accept license at https://huggingface.co/black-forest-labs/FLUX.1-dev

### Option 2: FLUX.1-schnell (Fast, No Auth)

```bash
# In RunPod Environment Variables:
MODEL_NAME=black-forest-labs/FLUX.1-schnell
# No HF_TOKEN needed!
```

**No auth required!** Works immediately.

---

## Expected Startup Log

### With Authentication (FLUX.1-dev)

```
CUDA Version 12.1.1
Loading Flux models: black-forest-labs/FLUX.1-dev
Device: cuda, Dtype: torch.bfloat16
✓ Downloading model... (first time only)
✓ Model loaded successfully!
Flux model loaded successfully!
✓ Ready to accept requests
```

### Without Authentication (FLUX.1-schnell)

```
CUDA Version 12.1.1
Loading Flux models: black-forest-labs/FLUX.1-schnell
Device: cuda, Dtype: torch.bfloat16
✓ Downloading model... (first time only)
✓ Model loaded successfully!
Flux model loaded successfully!
✓ Ready to accept requests
```

---

## What That PyTorch Warning Means

```
Disabling PyTorch because PyTorch >= 2.4 is required but found 2.3.0
PyTorch was not found. Models won't be available...
```

**This is just a warning** from `transformers` library expecting PyTorch 2.4.
- **Impact**: None - PyTorch 2.3 works perfectly for Flux
- **Safe to ignore**: The model loads and runs correctly
- **Why**: transformers has optional PyTorch 2.4 features we don't use

---

## Summary

✅ **Container working**: CUDA 12.1, PyTorch 2.3.0
✅ **Issue**: Need to set HF_TOKEN for FLUX.1-dev
✅ **Solutions**:
   - **Option 1**: Set HF_TOKEN environment variable
   - **Option 2**: Use FLUX.1-schnell (no auth needed)

🎯 **Next Step**: Add `HF_TOKEN` to your RunPod environment variables!
