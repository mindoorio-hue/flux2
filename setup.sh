#!/bin/bash
# Quick setup script for Flux2 endpoint development

set -e

echo "🚀 Setting up Flux2 Endpoint Development Environment"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "✓ Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
echo "✓ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "✓ Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "✓ Installing development dependencies..."
pip install pytest pytest-cov black flake8

# Create .env from example
if [ ! -f ".env" ]; then
    echo "✓ Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your HuggingFace token"
fi

# Create outputs directory
mkdir -p outputs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your HuggingFace token"
echo "  2. Run 'python local_test.py' to test locally"
echo "  3. Run 'docker build -t flux2-endpoint .' to build container"
echo ""
