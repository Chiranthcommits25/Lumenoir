# ---------------------------------------------------------------------------
# LUMENOIR backend image (API only)
#
# The frontend is a static build served separately; this image contains the
# FastAPI service and the full classical pipeline (NumPy + OpenCV are installed
# from requirements.txt). PyTorch and the Zero-DCE++ checkpoint are NOT bundled:
# classical mode works out of the box, while deep-learning and hybrid modes
# require installing requirements-inference.txt and provisioning the checkpoint.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
