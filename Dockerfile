# Dockerfile for SLAM3 Inference API
# Use this to deploy on Google Cloud Run, AWS ECS, or any container platform

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY slam3_pytorch.py .
COPY inference.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API server
CMD exec python inference.py api \
    --gcs-bucket ${GCS_BUCKET} \
    --gcs-checkpoint ${GCS_CHECKPOINT:-checkpoints/best.pt} \
    --host 0.0.0.0 \
    --port ${PORT}
