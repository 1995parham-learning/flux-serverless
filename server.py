import asyncio
import base64
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate(req: GenerateRequest):
    base_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "prompt": req.prompt,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "width": 1024,
            "height": 1024,
        }
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/run", json=payload, headers=headers, timeout=30
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        job_id = resp.json()["id"]

        while True:
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{base_url}/status/{job_id}", headers=headers, timeout=30
            )
            result = status_resp.json()
            status = result.get("status")

            if status == "COMPLETED":
                image_data = base64.b64decode(result["output"]["image_base64"])
                return Response(content=image_data, media_type="image/png")

            if status == "FAILED":
                raise HTTPException(status_code=500, detail=f"RunPod job failed: {result}")
