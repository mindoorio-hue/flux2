# How to Clear GitHub Actions Cache

## The Problem

Your first commit (`f34f594`) had the OLD Dockerfile with:
```dockerfile
FROM runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04
```

Even though we fixed it in commit `42df8b3` to:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

**GitHub Actions is still using cached layers from the first build!**

---

## Solution: Clear the Cache

### Method 1: Via GitHub Website (Easiest)

1. Go to: **https://github.com/mindoorio-hue/flux2/actions/caches**

2. Click **"Delete all caches"** button

3. Then go to: **https://github.com/mindoorio-hue/flux2/actions/workflows/docker.yml**

4. Click **"Run workflow"** → **"Run workflow"** button

---

### Method 2: Via GitHub CLI (After Installing)

```bash
cd /c/Users/mindoor/Documents/flux2-endpoint

# List all caches
gh cache list

# Delete all caches
gh cache delete --all

# Or delete specific cache
gh cache delete <cache-key>

# Trigger new build
git commit --allow-empty -m "Rebuild after cache clear"
git push
```

---

### Method 3: Run Clear Cache Workflow

1. Go to: **https://github.com/mindoorio-hue/flux2/actions/workflows/clear-cache.yml**

2. Click **"Run workflow"** → **"Run workflow"**

3. Wait for it to complete

4. Then manually trigger Docker Build workflow

---

## Verify the Fix

After clearing cache, check the new build:

1. **Go to Actions**: https://github.com/mindoorio-hue/flux2/actions

2. **Look for the "Verify Dockerfile" step** - should show:
   ```
   ✓ Correct base image found!
   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   ```

3. **Build should succeed** with pulling:
   ```
   [+] pulling pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
   ```

---

## Why This Happened

1. **First commit** pushed with wrong Dockerfile
2. **GitHub Actions** cached the `FROM` layer
3. **Even after fixing** Dockerfile, BuildKit used cached layer
4. **`no-cache: true`** doesn't always clear BASE image cache

---

## Prevention

From now on:
- ✅ Test Dockerfiles locally before committing
- ✅ Use `docker build --no-cache` for first build
- ✅ Clear cache after major base image changes

---

## Alternative: Force Rebuild with Different Tag

If clearing cache doesn't work, change the workflow to use a versioned tag:

```yaml
# In .github/workflows/docker.yml
tags: |
  type=ref,event=branch
  type=raw,value=v2  # Force new tag
```

Then in RunPod, use: `ghcr.io/mindoorio-hue/flux2:v2`

---

## Current Status

- ✅ Dockerfile fixed (commit 42df8b3)
- ✅ Workflow updated (commits 115aa8e, 8393189)
- ✅ Cache clearing workflow added (just now)
- ⏳ **NEED TO**: Clear cache and rebuild

---

## Steps Right Now

1. **Clear cache**: https://github.com/mindoorio-hue/flux2/actions/caches
2. **Trigger rebuild**: Push any commit or run workflow manually
3. **Monitor**: https://github.com/mindoorio-hue/flux2/actions
4. **Verify**: Build should succeed with correct base image

---

## Quick Command

```bash
# Nuclear option - force everything fresh
cd /c/Users/mindoor/Documents/flux2-endpoint
git commit --allow-empty -m "[FORCE BUILD] Clear all caches and rebuild"
git push origin main
```

Then go clear the cache on GitHub.

---

## Need Help?

If this STILL doesn't work after clearing cache:

1. Check workflow logs for the "Verify Dockerfile" step
2. Share the FULL error message
3. We can try alternative base images
