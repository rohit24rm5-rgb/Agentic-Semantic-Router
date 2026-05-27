FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for caching
COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling \
    && pip install --no-cache-dir .

# Copy application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "src.emotion_analysis.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
