FROM python:3.10-slim

# Install system dependencies required for OpenCV, git for CLIP, and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# ARGs for build-time environment injection
ARG QDRANT_URL
ARG QUERY_SERVICE_PORT
ARG CORS_ORIGINS
ARG CORS_ALLOW_CREDENTIALS

# Set ENVs for runtime
ENV QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
ENV QUERY_SERVICE_PORT=${QUERY_SERVICE_PORT:-5001}
ENV CORS_ORIGINS=${CORS_ORIGINS:-*}
ENV CORS_ALLOW_CREDENTIALS=${CORS_ALLOW_CREDENTIALS:-true}
ENV HOST=0.0.0.0

# Expose the configured port
EXPOSE ${QUERY_SERVICE_PORT}

# Run the FastAPI server via Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host $HOST --port $QUERY_SERVICE_PORT"]
