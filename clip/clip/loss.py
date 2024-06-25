import torch
import torch.nn as nn
from clip.constant import DEVICE


class CLIPLoss(nn.Module):
    def __init__(self, device = DEVICE):
        super().__init__()
        self.device = device
        self.img_loss = nn.CrossEntropyLoss()
        self.txt_loss = nn.CrossEntropyLoss()

    def forward(self, txt_log, img_log):
        # Loss function
        batch_size = txt_log.size(0)
        label = torch.arange(0, batch_size, dtype=torch.long, device=self.device)
        loss_images = self.img_loss(img_log, label)
        loss_text = self.txt_loss(txt_log, label)
        loss = (loss_images + loss_text) / 2
        return loss
