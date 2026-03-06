import argparse
import base64
import time
import requests


def run_sync(endpoint_id, api_key, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def run_async(endpoint_id, api_key, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"Job submitted: {job_id}")

    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    while True:
        time.sleep(5)
        resp = requests.get(status_url, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status")
        print(f"Status: {status}")
        if status == "COMPLETED":
            return result
        if status == "FAILED":
            print(f"Job failed: {result}")
            return result


def main():
    parser = argparse.ArgumentParser(description="Test RunPod FLUX.1-dev endpoint")
    parser.add_argument("--endpoint_id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api_key", required=True, help="RunPod API key")
    parser.add_argument("--prompt", default="A photo of a cat astronaut floating in space")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--async_mode", action="store_true", help="Use async /run endpoint")
    args = parser.parse_args()

    payload = {
        "input": {
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "width": args.width,
            "height": args.height,
        }
    }
    if args.seed is not None:
        payload["input"]["seed"] = args.seed

    print(f"Prompt: {args.prompt}")
    print("Sending request...")

    if args.async_mode:
        result = run_async(args.endpoint_id, args.api_key, payload)
    else:
        result = run_sync(args.endpoint_id, args.api_key, payload)

    if result.get("status") == "COMPLETED":
        output = result["output"]
        image_data = base64.b64decode(output["image_base64"])
        with open(args.output, "wb") as f:
            f.write(image_data)
        print(f"Image saved to {args.output}")
    else:
        print(f"Unexpected result: {result}")


if __name__ == "__main__":
    main()
