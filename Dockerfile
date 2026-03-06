FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download model weights at build time for faster cold starts
RUN uv run python3 -c "\
from huggingface_hub import login; \
login(token='${HF_TOKEN}'); \
from diffusers import FluxPipeline; \
FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')"

COPY handler.py .

CMD ["uv", "run", "python3", "-u", "handler.py"]
