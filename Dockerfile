# Multi-stage production container for PRISM Causal World Model
FROM python:3.12-slim

LABEL maintainer="PRISM Team"
LABEL description="PRISM: Causal World Models with Counterfactual Verification and Provable Safety Invariants"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

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
COPY .streamlit/ .streamlit/
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

# Default command launches the Streamlit operator console via scripts/run_dashboard.py
CMD ["python", "scripts/run_dashboard.py"]
