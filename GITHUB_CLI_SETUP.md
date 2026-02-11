# GitHub CLI Installation & Setup Guide

## Installation Status

✅ GitHub CLI installer downloaded to: `/tmp/gh-installer.msi`

---

## Complete Installation (Choose One Method)

### Method 1: Manual Install (Easiest)

1. **Run the installer:**
   - Double-click: `C:\tmp\gh-installer.msi`
   - Or from command line:
     ```powershell
     msiexec /i C:\tmp\gh-installer.msi
     ```

2. **Restart your terminal** (Git Bash, PowerShell, or CMD)

3. **Verify installation:**
   ```bash
   gh --version
   ```

---

### Method 2: Using Winget (Recommended)

```powershell
# Open PowerShell as Administrator
winget install --id GitHub.cli
```

---

### Method 3: Using Chocolatey

```powershell
# If you have Chocolatey installed
choco install gh
```

---

## After Installation

### 1. Authenticate with GitHub

```bash
gh auth login
```

Follow the prompts:
- What account? → **GitHub.com**
- Protocol? → **HTTPS**
- Authenticate? → **Login with a web browser**
- Copy the one-time code and press Enter
- Browser opens → paste code and authorize

### 2. Verify Authentication

```bash
gh auth status
```

Should show:
```
✓ Logged in to github.com as mindoorio-hue
✓ Git operations for github.com configured
✓ Token: *******************
```

---

## Useful GitHub CLI Commands for Your Project

### Check Repository Status
```bash
cd /c/Users/mindoor/Documents/flux2-endpoint
gh repo view
```

### View Issues
```bash
gh issue list
```

### View Pull Requests
```bash
gh pr list
gh pr view 1  # View PR #1
```

### Create Issue
```bash
gh issue create --title "Bug: Docker build failing" --body "Description here"
```

### View Workflows (GitHub Actions)
```bash
gh workflow list
gh run list
gh run view <run-id>
```

### View Dockerfile on GitHub
```bash
gh api repos/mindoorio-hue/flux2/contents/Dockerfile | jq -r '.content' | base64 -d
```

### Check Latest Commit
```bash
gh api repos/mindoorio-hue/flux2/commits/main | jq -r '.sha, .commit.message'
```

---

## Debug Your Docker Build Issue

Once GitHub CLI is installed, use these commands:

### 1. Verify Dockerfile on GitHub
```bash
cd /c/Users/mindoor/Documents/flux2-endpoint

# View Dockerfile from GitHub
gh api repos/mindoorio-hue/flux2/contents/Dockerfile --jq '.content' | base64 -d | head -5
```

Should show:
```
# Use official PyTorch CUDA image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
```

### 2. Check GitHub Actions Status
```bash
gh run list --limit 5
```

### 3. View Action Logs (if build failing in Actions)
```bash
gh run view --log
```

### 4. Force Workflow Re-run
```bash
gh workflow run docker.yml
```

---

## Troubleshooting

### "gh: command not found" after installation

**Solution 1: Restart terminal**
```bash
# Close and reopen Git Bash / PowerShell / CMD
```

**Solution 2: Add to PATH manually**
```bash
# Add to ~/.bashrc (Git Bash)
echo 'export PATH="/c/Program Files/GitHub CLI:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Solution 3: Use full path**
```bash
"/c/Program Files/GitHub CLI/gh.exe" --version
```

### "gh auth login" fails

**Solution**: Use token authentication
```bash
# Create token at: https://github.com/settings/tokens
# Then:
gh auth login --with-token < token.txt
```

---

## Alternative: Use GitHub API Directly

If GitHub CLI doesn't work, use curl:

### View Dockerfile
```bash
curl -H "Accept: application/vnd.github.v3.raw" \
  https://api.github.com/repos/mindoorio-hue/flux2/contents/Dockerfile
```

### View Latest Commit
```bash
curl https://api.github.com/repos/mindoorio-hue/flux2/commits/main | jq -r '.sha'
```

### Trigger Workflow (requires token)
```bash
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/mindoorio-hue/flux2/actions/workflows/docker.yml/dispatches \
  -d '{"ref":"main"}'
```

---

## Quick Start Commands

After installation and authentication:

```bash
# Go to project
cd /c/Users/mindoor/Documents/flux2-endpoint

# Check repo status
gh repo view

# View recent commits
gh log --limit 5

# Check workflows
gh workflow list

# View Dockerfile from GitHub (verify it's updated)
gh api repos/mindoorio-hue/flux2/contents/Dockerfile --jq '.content' | base64 -d | head -10
```

---

## Why Install GitHub CLI?

✅ Debug GitHub Actions workflows
✅ View files directly from GitHub
✅ Manage issues and PRs from terminal
✅ Trigger workflows manually
✅ Check repository status
✅ Verify what's actually on GitHub vs local

---

## Summary

1. **Install**: Run `C:\tmp\gh-installer.msi` or use `winget install --id GitHub.cli`
2. **Restart terminal**: Close and reopen Git Bash
3. **Authenticate**: `gh auth login`
4. **Test**: `gh repo view`

Once installed, you can debug your Docker build issue by verifying what's actually on GitHub!

---

## Current Docker Build Issue

Your Dockerfile is updated locally and on GitHub. If you're still getting the old image error:

1. **Verify GitHub has correct file:**
   ```bash
   gh api repos/mindoorio-hue/flux2/contents/Dockerfile --jq '.content' | base64 -d | head -5
   ```

2. **If building from RunPod**, use:
   ```
   https://github.com/mindoorio-hue/flux2.git#main
   ```

3. **If building locally**, ensure:
   - You're in correct directory
   - Docker Desktop is running
   - Use: `docker build -t flux2-endpoint .`
