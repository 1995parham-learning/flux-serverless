"""Deploy FLUX.1-dev as a RunPod serverless endpoint."""

import argparse
import json
import os

import requests

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "ghcr.io/1995parham-learning/flux-serverless:latest"
HF_TOKEN = os.environ["HF_TOKEN"]


def graphql_request(query, variables=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        print(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
        raise SystemExit(1)
    return data["data"]


def create_template(name, image):
    """Create a serverless template with the Docker image."""
    query = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
            isServerless
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            "imageName": image,
            "isServerless": True,
            "containerDiskInGb": 50,
            "volumeInGb": 0,
            "dockerArgs": "",
            "env": [
                {"key": "HF_TOKEN", "value": HF_TOKEN},
                {"key": "HF_HOME", "value": "/models"},
            ],
        }
    }
    data = graphql_request(query, variables)
    return data["saveTemplate"]


def create_endpoint(name, template_id, gpu_ids, min_workers=0, max_workers=1, idle_timeout=5):
    """Create a serverless endpoint using an existing template."""
    query = """
    mutation SaveEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
            gpuIds
            workersMin
            workersMax
            idleTimeout
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            "templateId": template_id,
            "gpuIds": gpu_ids,
            "workersMin": min_workers,
            "workersMax": max_workers,
            "idleTimeout": idle_timeout,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }
    data = graphql_request(query, variables)
    return data["saveEndpoint"]


def list_endpoints():
    query = """
    query Endpoints {
        myself {
            endpoints {
                id
                name
                gpuIds
                workersMin
                workersMax
                idleTimeout
                templateId
            }
        }
    }
    """
    data = graphql_request(query)
    return data["myself"]["endpoints"]


def delete_endpoint(endpoint_id):
    query = """
    mutation DeleteEndpoint($id: String!) {
        deleteEndpoint(id: $id)
    }
    """
    return graphql_request(query, {"id": endpoint_id})


def main():
    parser = argparse.ArgumentParser(description="Deploy FLUX.1-dev on RunPod")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = sub.add_parser("create", help="Create template + endpoint")
    create_parser.add_argument("--name", default="flux-serverless", help="Endpoint name")
    create_parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image")
    create_parser.add_argument(
        "--gpu",
        default="AMPERE_16",
        help="GPU type ID (default: AMPERE_16)",
    )
    create_parser.add_argument("--min-workers", type=int, default=0)
    create_parser.add_argument("--max-workers", type=int, default=1)
    create_parser.add_argument("--idle-timeout", type=int, default=5, help="Seconds before idle worker shuts down")

    # list
    sub.add_parser("list", help="List all serverless endpoints")

    # delete
    delete_parser = sub.add_parser("delete", help="Delete a serverless endpoint")
    delete_parser.add_argument("endpoint_id", help="Endpoint ID to delete")

    args = parser.parse_args()

    if args.command == "create":
        # Step 1: Create template
        template_name = f"{args.name}-template"
        print(f"Creating template '{template_name}' with image {args.image}...")
        template = create_template(template_name, args.image)
        template_id = template["id"]
        print(f"  Template ID: {template_id}")

        # Step 2: Create endpoint using the template
        print(f"Creating endpoint '{args.name}'...")
        result = create_endpoint(
            name=args.name,
            template_id=template_id,
            gpu_ids=args.gpu,
            min_workers=args.min_workers,
            max_workers=args.max_workers,
            idle_timeout=args.idle_timeout,
        )
        print(f"Endpoint created successfully!")
        print(f"  ID:          {result['id']}")
        print(f"  Name:        {result['name']}")
        print(f"  GPU:         {result['gpuIds']}")
        print(f"  Workers:     {result['workersMin']}-{result['workersMax']}")
        print()
        print("Test it with:")
        print(f"  uv run python test_endpoint.py --endpoint_id {result['id']} --prompt \"a cat in space\"")

    elif args.command == "list":
        endpoints = list_endpoints()
        if not endpoints:
            print("No endpoints found.")
            return
        for ep in endpoints:
            print(f"  {ep['id']}  {ep['name']}  gpu={ep['gpuIds']}  workers={ep['workersMin']}-{ep['workersMax']}")

    elif args.command == "delete":
        print(f"Deleting endpoint {args.endpoint_id}...")
        delete_endpoint(args.endpoint_id)
        print("Deleted.")


if __name__ == "__main__":
    main()
