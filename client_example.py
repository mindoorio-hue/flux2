"""
Example client for Flux2 RunPod endpoint
Demonstrates how to call the deployed endpoint and save results
"""
import requests
import base64
import json
import time
from pathlib import Path
from PIL import Image
from io import BytesIO


class Flux2Client:
    """Client for Flux2 RunPod endpoint"""

    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize client

        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: int = None,
        output_format: str = "PNG",
        timeout: int = 300
    ):
        """
        Generate images using the Flux2 endpoint

        Args:
            prompt: Text description
            negative_prompt: What to avoid
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            num_images: Number of images to generate
            seed: Random seed
            output_format: Output format (PNG, JPEG, WEBP)
            timeout: Request timeout in seconds

        Returns:
            dict: Response with images and metadata
        """
        payload = {
            "input": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "output_format": output_format
            }
        }

        if seed is not None:
            payload["input"]["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        print(f"Sending request to {self.base_url}/runsync")
        print(f"Prompt: {prompt[:50]}...")

        start_time = time.time()

        # Use runsync for synchronous request
        response = requests.post(
            f"{self.base_url}/runsync",
            headers=headers,
            json=payload,
            timeout=timeout
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        result = response.json()

        print(f"✅ Request completed in {elapsed:.2f}s")

        return result

    def generate_async(self, **kwargs):
        """
        Generate images asynchronously (for long-running requests)

        Returns:
            str: Job ID for status checking
        """
        payload = {"input": {k: v for k, v in kwargs.items() if v is not None}}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}/run",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        return response.json()["id"]

    def check_status(self, job_id: str):
        """Check status of async job"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=headers
        )

        return response.json()

    @staticmethod
    def save_images(result: dict, output_dir: str = "outputs"):
        """
        Save images from result to disk

        Args:
            result: Response from generate()
            output_dir: Directory to save images

        Returns:
            list: Paths to saved images
        """
        Path(output_dir).mkdir(exist_ok=True)

        if result.get("status") != "success":
            raise Exception(f"Generation failed: {result.get('error')}")

        output = result.get("output", result)
        images_b64 = output.get("images", [])
        metadata = output.get("metadata", {})

        saved_paths = []

        for i, img_b64 in enumerate(images_b64):
            # Decode base64
            img_data = base64.b64decode(img_b64)
            img = Image.open(BytesIO(img_data))

            # Generate filename
            seed = metadata.get("seed", "unknown")
            prompt_slug = metadata.get("prompt", "image")[:30].replace(" ", "_")
            filename = f"{prompt_slug}_seed{seed}_{i}.png"
            filepath = Path(output_dir) / filename

            # Save
            img.save(filepath)
            saved_paths.append(str(filepath))

            print(f"💾 Saved: {filepath}")

        return saved_paths


# Example usage
if __name__ == "__main__":
    # Configuration
    ENDPOINT_ID = "your-endpoint-id-here"
    API_KEY = "your-api-key-here"

    # Initialize client
    client = Flux2Client(ENDPOINT_ID, API_KEY)

    # Generate image
    try:
        result = client.generate(
            prompt="A majestic dragon flying over a medieval castle at sunset, highly detailed, fantasy art",
            negative_prompt="blurry, low quality, distorted",
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images=1,
            seed=42
        )

        # Save images
        paths = client.save_images(result)

        print(f"\n✅ Generated {len(paths)} image(s)")
        print(f"Inference time: {result['output']['metadata']['inference_time']}s")
        print(f"Seed: {result['output']['metadata']['seed']}")

    except Exception as e:
        print(f"❌ Error: {e}")
