# Dockerfile
# Production Docker configuration for cloud deployment

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_TYPE=yolov3

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# Copy the rest of the application codebase
COPY . .

# Run weight downloader during image build to bake YOLOv3-tiny into the image
# This prevents cold-start download delays and Render service timeouts
RUN python download_weights.py

# Expose Render standard port
EXPOSE 10000

# Run FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
