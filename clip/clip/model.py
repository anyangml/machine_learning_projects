import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from clip.loss import CLIPLoss

class CLIP(nn.Module):
    def __init__(self, txt_encoder, img_encoder, embd_dim, temperature):
        super().__init__()

        assert 0 < temperature, "temperature must be greater than zero."
        self.temperature = temperature
        self.embd_dim = embd_dim

        self.txt_encoder = txt_encoder
        self.img_encoder = img_encoder

        self.txt_proj = nn.Linear(
            self.txt_encoder.config.out_dim, self.embd_dim, bias=False
        )
        self.img_proj = nn.Linear(
            self.img_encoder.config.out_dim, self.embd_dim, bias=False
        )
        self.temperature = nn.Parameter(torch.log(torch.tensor(1/temperature)))

    def forward(self, text, image):
        encoded_text = self.txt_encoder(text)
        encoded_image = self.img_encoder(image)

        embd_text = F.normalize(self.txt_proj(encoded_text), p=2, dim=1)  # L2 norm (B, D)
        embd_image = F.normalize(self.img_proj(encoded_image), p=2, dim=1)  # L2 norm (B, D)

        # scaled pairwise cosine similarities (B, B)
        logits = torch.mm(embd_text, embd_image.T) * torch.clamp(torch.exp(self.temperature), min=0.01, max=100.0)

        return logits, logits.T  # text, image

class CLIPWrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, text, image):
        txt_log, img_log = self.model(text, image)
        loss = self.loss_fn(txt_log, img_log)

        return loss