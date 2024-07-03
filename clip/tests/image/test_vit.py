from clip.image.vit import ViT
from clip.image.vit import ViTConfig
import torch


def test_ViT_forward_shape():
    config = ViTConfig(device=torch.device("cpu"))
    vit = ViT(config)
    dummy_imgs = torch.randn(2, 3, 112, 112)
    encoded = vit(dummy_imgs)
    assert encoded.shape == (2, 768)
