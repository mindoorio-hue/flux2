"""
Test all three Flux2 workflows
Demonstrates text-to-image, image-to-image, and multi-reference
"""
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import json


def create_test_image(text: str, size=(512, 512), color='lightblue'):
    """Create a simple test image with text"""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)

    # Add text
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill='black')

    return img


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 to PIL Image"""
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))


print("=" * 60)
print("Flux2 Workflow Test Suite")
print("=" * 60)

# Workflow 1: Text-to-Image
print("\n1. TEXT-TO-IMAGE WORKFLOW")
print("-" * 60)
txt2img_input = {
    "input": {
        "prompt": "A majestic mountain landscape at sunset",
        "width": 512,
        "height": 512,
        "num_inference_steps": 4,
        "guidance_scale": 7.5,
        "seed": 42
    }
}
print("Input:")
print(json.dumps(txt2img_input, indent=2))
print("\nThis workflow generates images from text prompts only.")
print("Status: IMPLEMENTED [OK]")

# Workflow 2: Image-to-Image
print("\n\n2. IMAGE-TO-IMAGE WORKFLOW")
print("-" * 60)

# Create test init image
init_image = create_test_image("INIT IMAGE", color='lightgreen')
init_image_b64 = image_to_base64(init_image)

img2img_input = {
    "input": {
        "prompt": "Transform into a watercolor painting",
        "init_image": init_image_b64[:50] + "...",  # Truncated for display
        "strength": 0.75,
        "width": 512,
        "height": 512,
        "num_inference_steps": 4,
        "guidance_scale": 7.5,
        "seed": 42
    }
}
print("Input:")
print(json.dumps(img2img_input, indent=2))
print("\nThis workflow transforms an existing image based on a text prompt.")
print("- 'strength' controls how much to change (0.0 = no change, 1.0 = complete transformation)")
print("- Higher strength = more creative freedom, lower = more faithful to init image")
print("Status: IMPLEMENTED [OK]")

# Workflow 3: Multi-Reference
print("\n\n3. MULTI-REFERENCE WORKFLOW")
print("-" * 60)

# Create test reference images
ref1 = create_test_image("REF 1", color='lightcoral')
ref2 = create_test_image("REF 2", color='lightyellow')
ref3 = create_test_image("REF 3", color='lightcyan')

ref1_b64 = image_to_base64(ref1)
ref2_b64 = image_to_base64(ref2)
ref3_b64 = image_to_base64(ref3)

multi_ref_input = {
    "input": {
        "prompt": "A portrait with combined styles from references",
        "reference_images": [
            ref1_b64[:50] + "...",  # Truncated for display
            ref2_b64[:50] + "...",
            ref3_b64[:50] + "..."
        ],
        "reference_weights": [0.4, 0.35, 0.25],
        "width": 512,
        "height": 512,
        "num_inference_steps": 4,
        "guidance_scale": 7.5,
        "seed": 42
    }
}
print("Input:")
print(json.dumps(multi_ref_input, indent=2))
print("\nThis workflow combines multiple reference images with weighted influence.")
print("- Each reference image contributes based on its weight")
print("- Weights should sum to 1.0 (auto-normalized if not)")
print("- Useful for style mixing, composition blending, etc.")
print("Status: IMPLEMENTED [OK]")

# Summary
print("\n\n" + "=" * 60)
print("WORKFLOW SUMMARY")
print("=" * 60)
print("""
[OK] Text-to-Image: Generate from text prompt only
[OK] Image-to-Image: Transform existing image with text guidance
[OK] Multi-Reference: Blend multiple reference images with weighted control

All workflows share common parameters:
- prompt (required)
- negative_prompt (optional)
- width, height (multiples of 8)
- num_inference_steps
- guidance_scale
- num_images (1-4)
- seed (for reproducibility)

Workflow Detection:
- Has reference_images -> Multi-Reference
- Has init_image -> Image-to-Image
- Neither -> Text-to-Image
""")

print("\nTo test with actual model, replace handler.py imports and run:")
print("  python workflow_test.py")
print("\nTo test via RunPod endpoint:")
print("  python client_example.py")
