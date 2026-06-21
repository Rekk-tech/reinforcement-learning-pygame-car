# ── Dockerfile ──────────────────────────────────────────────────────
# Multi-stage build cho Deep Learning Cars
#
# Build:   docker build -t deep-learning-cars .
# Run GA:  docker run deep-learning-cars --headless --generations 50
# Run PPO: docker run deep-learning-cars --algorithm ppo --headless --generations 100
# ─────────────────────────────────────────────────────────────────────

# Stage 1: Base image
FROM python:3.13-slim AS base

WORKDIR /app

# System dependencies (SDL cho pygame headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM deps AS app

# Copy source code
COPY main.py .
COPY mlflow_tracking.py .
COPY configs/ ./configs/
COPY src/ ./src/

# Tao thu muc checkpoints
RUN mkdir -p checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy
ENV SDL_AUDIODRIVER=dummy

# Default: chay headless GA
ENTRYPOINT ["python", "main.py"]
CMD ["--headless", "--generations", "50"]
