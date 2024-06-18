import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.label = torch.arange(0, self.batch_size)

    def forward(self, img_log, txt_log):
        img_loss = nn.CrossEntropyLoss()
        txt_loss = nn.CrossEntropyLoss()

        # Loss function
        loss_images = img_loss(img_log, self.label)
        loss_text = txt_loss(txt_log, self.label)
        loss = (loss_images + loss_text) / 2
        return loss
