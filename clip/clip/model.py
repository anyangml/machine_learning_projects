import torch
from torch import nn
from constant import DEVICE

class CLIP(nn.Module):
    def __init__(self, txt_encoder, img_encoder):
        super(CLIP, self).__init__()
        self.txt_encoder = txt_encoder
        self.img_encoder = img_encoder

    def forward(self, text, image):
        encoded_text = self.txt_encoder(text)
        encoded_image = self.img_encoder(image)

