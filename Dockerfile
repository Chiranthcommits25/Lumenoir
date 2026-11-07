# ---------------------------------------------------------------------------
# LUMENOIR backend image (API only)
#
# The frontend is a static build served separately; this image contains the
# FastAPI service, the full classical pipeline (NumPy + OpenCV), PyTorch and
# the Zero-DCE++ checkpoint so all three enhancement modes work on Railway
# (CPU-only). PyTorch and the checkpoint are installed at build time because
# the checkpoint is git-ignored and must not be committed.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LUMENOIR_MODEL_ID=zero_dce_plus_plus \
    LUMENOIR_INFERENCE_DEVICE=cpu \
    LUMENOIR_LOAD_MODEL_ON_STARTUP=true

WORKDIR /app

# System deps for downloading the checkpoint
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Runtime + inference dependencies (fastapi, uvicorn, pillow, numpy, opencv,
# torch). requirements-inference.txt includes requirements.txt via -r.
COPY requirements.txt requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

# Download and verify the Zero-DCE++ checkpoint (git-ignored locally)
ARG ZERODCE_CHECKPOINT_URL="https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
ARG ZERODCE_CHECKPOINT_SHA256="ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84"

RUN mkdir -p backend/weights && \
    curl -fsSL -o backend/weights/zero_dce_plus_plus.pth "${ZERODCE_CHECKPOINT_URL}" && \
    echo "${ZERODCE_CHECKPOINT_SHA256}  backend/weights/zero_dce_plus_plus.pth" | sha256sum -c -

# Backend source
COPY backend ./backend

EXPOSE 8000

# Railway injects $PORT; listen on it
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
