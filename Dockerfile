# Multi-stage production container for PRISM Causal World Model
FROM python:3.11-slim

LABEL maintainer="PRISM Team"
LABEL description="PRISM: Causal World Models with Counterfactual Verification and Provable Safety Invariants"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements first for layer caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY prism/ prism/
COPY configs/ configs/
COPY artifacts/ artifacts/
COPY data/ data/
COPY scripts/ scripts/
COPY tests/ tests/

# Install PRISM in editable mode
RUN pip install -e .

# Expose Streamlit (8501) and Web SPA (8000)
EXPOSE 8501 8000

# Default command launches the Streamlit operator console
CMD ["streamlit", "run", "scripts/run_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
