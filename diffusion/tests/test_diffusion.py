from diffusion.diffusion_wrap import Diffusion
from diffusion.unet import UNetWithTime
import torch

def test_diffusion():
    model = Diffusion(UNetWithTime(),"cpu")
    x = torch.randn(2, 1, 572, 572)
    y = model(x)
    assert y.shape != x.shape