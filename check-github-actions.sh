#!/bin/bash
# Check GitHub Actions status

echo "=== Checking GitHub Actions Status ==="
echo ""

echo "Latest commit on GitHub:"
curl -s https://api.github.com/repos/mindoorio-hue/flux2/commits/main | grep -A 1 '"sha"' | head -2

echo ""
echo "Dockerfile on GitHub (first 5 lines):"
curl -s https://raw.githubusercontent.com/mindoorio-hue/flux2/main/Dockerfile | head -5

echo ""
echo "=== Recent workflow runs ==="
curl -s https://api.github.com/repos/mindoorio-hue/flux2/actions/runs | grep -E '"name"|"status"|"conclusion"|"created_at"' | head -20

echo ""
echo "To see full logs, go to:"
echo "https://github.com/mindoorio-hue/flux2/actions"
