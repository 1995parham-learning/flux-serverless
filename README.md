# Flux Serverless

Serverless text-to-image generation endpoint powered by FLUX.1-dev on RunPod. Send a text prompt, get a high-quality 1024x1024 image back.

## Overview

This project deploys the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model by Black Forest Labs as a [RunPod](https://www.runpod.io/) serverless endpoint. The endpoint accepts a text prompt and returns a generated image as a base64-encoded PNG.

## How It Works

The architecture is straightforward:

1. **Handler (`handler.py`)** — A Python script that acts as the serverless worker. When the container starts, it loads the FLUX.1-dev pipeline into GPU memory using the `diffusers` library. The model stays loaded and serves incoming requests. Each request provides a text prompt (and optional parameters like image size or seed), the handler runs inference through the pipeline, and returns the generated image encoded as a base64 string.

2. **Docker Image (`Dockerfile`)** — Packages the handler, dependencies, and pre-downloaded model weights into a single container image based on `nvidia/cuda:12.1.1`. The model weights (~30 GB) are baked into the image at build time so that cold starts only need to load the model into VRAM rather than downloading it from Hugging Face first.

3. **RunPod Serverless** — RunPod pulls the Docker image and spins up GPU workers on demand. When a request arrives at the endpoint URL, RunPod routes it to an available worker (or starts a new one). Workers scale to zero when idle, so you only pay for actual compute time.

```
User Request (prompt)
        |
        v
  RunPod API Gateway
        |
        v
  GPU Worker (Docker container)
        |
        v
  FLUX.1-dev Pipeline (diffusers)
        |
        v
  Base64 PNG Response
```

## Server

`server.py` is a FastAPI server that acts as a local proxy to the RunPod endpoint. It exposes a `POST /generate` endpoint that accepts a JSON body with a `prompt` field, submits the job to RunPod, polls for completion, and returns the generated image directly as a PNG response.

The server reads `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` from environment variables:

```bash
export RUNPOD_API_KEY="<your-runpod-api-key>"
export RUNPOD_ENDPOINT_ID="<your-endpoint-id>"
uvicorn server:app --reload
```

## Project Structure

```
.
├── README.md
├── handler.py          # RunPod serverless handler
├── server.py           # FastAPI server proxying requests to RunPod
├── Dockerfile          # Docker image definition
├── deploy.py           # Deploy/manage RunPod endpoints via API
├── pyproject.toml      # Python dependencies (uv)
└── test_endpoint.py    # Script to test the deployed endpoint
```

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with GPU credits
- [Docker](https://www.docker.com/) installed locally
- A [Hugging Face](https://huggingface.co/) account with access to the FLUX.1-dev model
- Python 3.10+

## Step-by-Step Setup

### Step 1 — RunPod Account

1. Create an account at [runpod.io](https://www.runpod.io/).
2. Share your account email with `hailong.yang@runpod.io` to receive free credits.
3. Once credits are applied, grab your **API Key** from the RunPod console under **Settings > API Keys**.

### Step 2 — Hugging Face Token

FLUX.1-dev is a gated model, so you need explicit access:

1. Go to [FLUX.1-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev) and accept the license agreement.
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `read` permission.

### Step 2.5 — Set Environment Variables

All scripts read tokens from environment variables. Export them before running any commands:

```bash
export RUNPOD_API_KEY="<your-runpod-api-key>"
export HF_TOKEN="<your-hugging-face-token>"
export RUNPOD_ENDPOINT_ID="<your-endpoint-id>"
```

### Step 3 — Build the Docker Image

The build downloads the full model weights, so it requires ~30 GB of disk space and a good internet connection.

```bash
docker build -t <your-dockerhub-username>/runpod-flux:latest \
  --build-arg HF_TOKEN=$HF_TOKEN .
```

**What happens during build:**
- Installs a CUDA 12.1 runtime base image
- Installs Python packages from `requirements.txt` (PyTorch, diffusers, transformers, runpod SDK, etc.)
- Downloads FLUX.1-dev model weights from Hugging Face and caches them inside the image
- Copies `handler.py` as the container entrypoint

### Step 4 — Push to Docker Hub

```bash
docker login
docker push <your-dockerhub-username>/runpod-flux:latest
```

### Step 5 — Deploy on RunPod

**Option A — Using the deploy script (recommended):**

```bash
pip install requests

# Create endpoint
python deploy.py create --image ghcr.io/1995parham-learning/flux-serverless:latest

# List endpoints
python deploy.py list

# Delete an endpoint
python deploy.py delete <ENDPOINT_ID>
```

**Option B — Manual setup via RunPod console:**

1. Go to the [RunPod Serverless Console](https://www.runpod.io/console/serverless).
2. Click **New Endpoint**.
3. Configure:
   - **Container Image**: `ghcr.io/1995parham-learning/flux-serverless:latest`
   - **GPU Type**: Select a GPU with at least 24 GB VRAM (e.g., NVIDIA A100, RTX 4090, or RTX A6000)
   - **Active Workers**: Set minimum to 0 (scale to zero when idle)
   - **Max Workers**: Set based on expected concurrency (1 is fine for testing)
   - **Environment Variables**:
     - `HF_TOKEN` = `<your-hugging-face-token>`
     - `HF_HOME` = `/models`
4. Click **Deploy**.
5. Note the **Endpoint ID** shown on the dashboard — you'll need it for API calls.

### Step 6 — Test the Endpoint

Once the endpoint is deployed and shows as "Ready":

**Using the included test script:**

```bash
pip install requests

python test_endpoint.py \
  --endpoint_id <ENDPOINT_ID> \
  --prompt "A photo of a cat astronaut floating in space"
```

This sends the request, waits for completion, and saves the result to `output.png`.

**Using cURL:**

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A photo of a cat astronaut floating in space"
    }
  }'
```

The first request may take longer due to cold start (loading model into VRAM). Subsequent requests while the worker is warm will be significantly faster.

## API Reference

### Request Format

```json
{
  "input": {
    "prompt": "A photo of a cat astronaut floating in space",
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }
}
```

| Parameter              | Type   | Default | Description                                      |
|------------------------|--------|---------|--------------------------------------------------|
| `prompt`               | string | —       | **(Required)** Text description of the image      |
| `num_inference_steps`  | int    | 28      | Number of denoising steps (higher = better quality, slower) |
| `guidance_scale`       | float  | 3.5     | How closely to follow the prompt (higher = more literal) |
| `width`                | int    | 1024    | Output image width in pixels                      |
| `height`               | int    | 1024    | Output image height in pixels                     |
| `seed`                 | int    | random  | Seed for reproducible generation                  |

### Response Format

```json
{
  "output": {
    "image_base64": "<base64-encoded PNG image>",
    "seed": 42
  }
}
```

To decode the image, base64-decode the `image_base64` field and save it as a `.png` file.

## Resources

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/get-started)
- [FLUX.1-dev Model Card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [Diffusers FluxPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
