import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CLIP(nn.Module):
    def __init__(self, txt_encoder, img_encoder, embd_dim, temperature):
        super().__init__()

        assert 0 <= temperature <= 1, "temperature must be in range [0,1]"
        self.temperature = temperature
        self.embd_dim = embd_dim

        self.txt_encoder = txt_encoder
        self.img_encoder = img_encoder

        self.txt_proj = nn.Linear(
            self.txt_encoder.config.vocab_size, self.embd_dim, bias=False
        )
        self.img_proj = nn.Linear(
            self.img_encoder.config.out_dim, self.embd_dim, bias=False
        )

    def forward(self, text, image):
        encoded_text = self.txt_encoder(text)
        encoded_image = self.img_encoder(image)

        embd_text = F.normalize(self.txt_proj(encoded_text), dim=1)  # (B, D)
        embd_image = F.normalize(self.img_proj(encoded_image), dim=1)  # (B, D)

        # scaled pairwise cosine similarities (B, B)
        logits = torch.mm(embd_text, embd_image.T) * np.exp(self.temperature)

        return logits, logits.T  # text, image
