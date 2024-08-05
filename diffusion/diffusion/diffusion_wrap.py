import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNetWithTime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion(nn.Module):
    def __init__(self, model: UNetWithTime, timesteps: int=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0: torch.Tensor):

        noise = torch.randn_like(x0)
        t = torch.randint(0, self.timesteps, (noise.shape[0],)).long().to(DEVICE)
        x_t = x0 * torch.sqrt(self.alpha_bar[t].view(-1,1,1,1)) + noise * torch.sqrt(1 - self.alpha_bar[t].view(-1,1,1,1))
        pred_noise = self.model(x_t, t)
        return F.mse_loss(noise, pred_noise)
    
