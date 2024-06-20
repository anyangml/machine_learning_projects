import pytest
import json
from unittest.mock import patch
from torchvision import transforms
from torchvision.io import read_image
import tiktoken
import tempfile
import shutil
import os
import torch
import numpy as np
from PIL import Image
from clip.constant import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_SEQ_LENGTH


@pytest.fixture
def mock_data():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Mock JSON data
    json_data = [
        {"key": "image1", "caption": "A caption for image 1", "error_message": None},
        {"key": "image2", "caption": "A caption for image 2", "error_message": None},
    ]

    # Write mock JSON files
    for i, data in enumerate(json_data):
        json_path = os.path.join(temp_dir, f"data_{i}.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

    # Write mock image files
    for data in json_data:
        img_path = os.path.join(temp_dir, f"{data['key']}.jpg")
        mock_img = mock_image_data()
        mock_img.save(img_path)

    # Mock image reading
    patch("clip.data.dataset.CLIPDataset.read_img", side_effect=mock_read_img).start()

    # Mock text tokenization
    patch("clip.data.dataset.CLIPDataset.tokenize", side_effect=mock_tokenize).start()

    yield temp_dir

    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    patch.stopall()


def mock_read_img(img_path):
    return transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True)(
        read_image(img_path)
    )


def mock_image_data():
    img_array = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


def mock_tokenize(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    if len(tokens) >= MAX_SEQ_LENGTH:
        tokens = tokens[: MAX_SEQ_LENGTH - 1]
        tokens += [tokenizer._special_tokens["<|endoftext|>"]]
    else:
        # padding after EOS token for batch process
        tokens += [tokenizer._special_tokens["<|endoftext|>"]]
        tokens += [0] * (
            MAX_SEQ_LENGTH - len(tokens)
        )

    return torch.tensor(tokens,dtype=torch.int32)
