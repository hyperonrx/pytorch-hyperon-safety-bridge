# Use Python 3.10 (Stable for PyTorch + Hyperon)
FROM python:3.10-slim

# Install system dependencies (needed for compiling some ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# We install hyperon explicitly from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create directories for data and artifacts
RUN mkdir -p subjects ohio_demo artifacts_optimization_run

# Default command
CMD ["python", "pancreas_hyperon.py"]