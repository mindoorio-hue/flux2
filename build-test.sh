#!/bin/bash
# Test build script to verify Dockerfile is correct

echo "=== Checking Dockerfile ==="
echo "First line of Dockerfile:"
head -1 Dockerfile
echo ""

echo "=== Building Docker image ==="
echo "Building with: docker build -t flux2-endpoint:test ."
echo ""

# Uncomment to actually build:
# docker build -t flux2-endpoint:test .

echo "To build, run:"
echo "  docker build -t flux2-endpoint ."
echo ""
echo "Or with specific Dockerfile:"
echo "  docker build -f Dockerfile.runpod -t flux2-endpoint ."
