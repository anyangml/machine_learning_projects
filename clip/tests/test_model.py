from clip.image.vit import ViT, ViTConfig
from clip.languange.gpt import GPT, GPTConfig
from clip.model import CLIP
import torch


def test_clip_forward():
    img_encoder = ViT(ViTConfig(device=torch.device("cpu")))
    text_encoder = GPT(GPTConfig(device=torch.device("cpu")))

    clip = CLIP(
        txt_encoder=text_encoder, img_encoder=img_encoder, embd_dim=512, temperature=0.5
    )

    dummy_img = torch.randn(
        1, ViTConfig.img_channel, ViTConfig.img_height, ViTConfig.img_width
    )
    dummy_txt = torch.randint(0, GPTConfig.vocab_size, (1, 77))
    dummy_txt[0, -1] = 50256  # add EOT

    txt_log, img_log = clip(dummy_txt, dummy_img)
    torch.testing.assert_close(txt_log, img_log.T)
    assert txt_log.shape == (1, 1)
