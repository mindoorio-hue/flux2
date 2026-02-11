# Use official PyTorch CUDA image with PyTorch 2.5+ (required for Flux)
# Flux requires torch.nn.RMSNorm (PyTorch 2.4.0+) and enable_gqa (PyTorch 2.5.0+)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models (optional - comment out to download on first run)
# ENV HF_HOME=/app/models
# RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')"

# Copy application code
COPY handler.py .
COPY src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=black-forest-labs/FLUX.2-dev

# Start the handler
CMD ["python", "-u", "handler.py"]
