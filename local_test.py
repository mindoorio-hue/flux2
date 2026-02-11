"""
Local testing script for Flux2 handler
Run this to test the handler locally before deploying
"""
import json
import base64
from handler import handler
from PIL import Image
from io import BytesIO
import os


def save_base64_image(b64_string: str, output_path: str):
    """Save base64 encoded image to file"""
    img_data = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_data))
    img.save(output_path)
    print(f"Image saved to: {output_path}")


def test_handler():
    """Test the handler with sample input"""

    # Load test input
    with open("test_input.json", "r") as f:
        test_event = json.load(f)

    print("Testing Flux2 handler...")
    print(f"Prompt: {test_event['input']['prompt']}")

    # Call handler
    result = handler(test_event)

    # Check result
    if result["status"] == "success":
        print(f"\n✅ Success!")
        print(f"Generated {len(result['images'])} image(s)")
        print(f"Inference time: {result['metadata']['inference_time']}s")
        print(f"Seed: {result['metadata']['seed']}")

        # Save images
        os.makedirs("outputs", exist_ok=True)
        for i, img_b64 in enumerate(result["images"]):
            output_path = f"outputs/test_image_{i}.png"
            save_base64_image(img_b64, output_path)

        return True
    else:
        print(f"\n❌ Error: {result['error']}")
        if "traceback" in result:
            print(f"\nTraceback:\n{result['traceback']}")
        return False


if __name__ == "__main__":
    success = test_handler()
    exit(0 if success else 1)
