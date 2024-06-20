from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from pathlib import Path
import os
import json
from clip.constant import MAX_SEQ_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH
import tiktoken
from typing import List
import logging

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class CLIPDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_len: int = MAX_SEQ_LENGTH,
        img_height: int = IMAGE_HEIGHT,
        img_width: int = IMAGE_WIDTH,
    ):
        self.data_dir = data_dir
        self.max_len = max_len
        self.img_height = img_height
        self.img_width = img_width

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.metadata = [self.read_json(js) for js in Path(data_dir).rglob("*.json")]
        self.img_data = [
            self.read_img(os.path.join(data_dir, f"{data['key']}.jpg"))
            for data in self.metadata
            if not data["error_message"]
        ]
        self.txt_data = [
            self.tokenize(data["caption"])
            for data in self.metadata
            if not data["error_message"]
        ]

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        return self.img_data[idx], self.txt_data[idx]

    def read_json(self, js):
        with open(js, "r") as f:
            data = json.load(f)
        return data

    def read_img(self, img_path):
        raw_img = read_image(img_path)
        logger.info(
            f"Resizing input image from {raw_img.shape[1:]} to {(self.img_height, self.img_width)}"
        )
        return transforms.Resize((self.img_height, self.img_width), antialias=True)(
            raw_img
        )

    def tokenize(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode(text)
        if len(tokens) >= self.max_len:
            tokens = tokens[: self.max_len - 1]
        return tokens + [self.tokenizer._special_tokens["<|endoftext|>"]]
