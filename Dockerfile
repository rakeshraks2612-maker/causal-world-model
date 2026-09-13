# Production container for PRISM Web Dashboard
FROM python:3.12-slim

LABEL maintainer="PRISM Team"
LABEL description="PRISM: Causal World Models with Counterfactual Verification and Provable Safety Invariants"

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

# Patch Streamlit static assets with official PRISM logo
RUN python -c "\
import streamlit, shutil; \
from pathlib import Path; \
st_s = Path(streamlit.__file__).parent / 'static'; \
[shutil.copyfile(f, st_s / f.name) for f in Path('prism/dashboard/web').glob('favicon.*') if st_s.exists()]"

# Expose ports
EXPOSE 8000 10000

# Default command launches the PRISM Web SPA directly on $PORT
CMD ["python", "scripts/run_web.py"]
