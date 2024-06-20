import torch
import torch.nn as nn
from clip.constant import DEVICE


class CLIPLoss(nn.Module):
    def __init__(self, batch_size: int, device = DEVICE):
        super().__init__()
        self.batch_size = batch_size
        self.label = torch.arange(0, self.batch_size, dtype=torch.long, device=device)
        self.img_loss = nn.CrossEntropyLoss()
        self.txt_loss = nn.CrossEntropyLoss()

    def forward(self, txt_log, img_log):
        # Loss function
        loss_images = self.img_loss(img_log, self.label)
        loss_text = self.txt_loss(txt_log, self.label)
        loss = (loss_images + loss_text) / 2
        return loss
