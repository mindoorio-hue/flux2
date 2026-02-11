"""
Unit tests for Flux2 handler
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from handler import handler, image_to_base64
from PIL import Image


def test_image_to_base64():
    """Test image to base64 conversion"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    b64_str = image_to_base64(img)

    assert isinstance(b64_str, str)
    assert len(b64_str) > 0


def test_handler_missing_prompt():
    """Test handler with missing prompt"""
    event = {"input": {}}

    result = handler(event)

    assert result["status"] == "error"
    assert "prompt" in result["error"].lower()


def test_handler_invalid_dimensions():
    """Test handler with invalid dimensions (not multiple of 8)"""
    event = {
        "input": {
            "prompt": "test prompt",
            "width": 1023,  # Not multiple of 8
            "height": 1024
        }
    }

    result = handler(event)

    assert result["status"] == "error"
    assert "multiple of 8" in result["error"]


def test_handler_invalid_num_images():
    """Test handler with invalid number of images"""
    event = {
        "input": {
            "prompt": "test prompt",
            "num_images": 10  # Too many
        }
    }

    result = handler(event)

    assert result["status"] == "error"
    assert "num_images" in result["error"]


@patch('handler.pipe')
def test_handler_success(mock_pipe):
    """Test successful image generation"""
    # Mock the pipeline output
    mock_image = Image.new('RGB', (1024, 1024), color='blue')
    mock_result = MagicMock()
    mock_result.images = [mock_image]
    mock_pipe.return_value = mock_result

    event = {
        "input": {
            "prompt": "a beautiful landscape",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 4,
            "seed": 42
        }
    }

    result = handler(event)

    assert result["status"] == "success"
    assert len(result["images"]) == 1
    assert "metadata" in result
    assert result["metadata"]["seed"] == 42
    assert result["metadata"]["prompt"] == "a beautiful landscape"


@patch('handler.pipe')
def test_handler_multiple_images(mock_pipe):
    """Test generating multiple images"""
    # Mock multiple images
    mock_images = [
        Image.new('RGB', (512, 512), color='red'),
        Image.new('RGB', (512, 512), color='green')
    ]
    mock_result = MagicMock()
    mock_result.images = mock_images
    mock_pipe.return_value = mock_result

    event = {
        "input": {
            "prompt": "test",
            "num_images": 2,
            "width": 512,
            "height": 512
        }
    }

    result = handler(event)

    assert result["status"] == "success"
    assert len(result["images"]) == 2
