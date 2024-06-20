import torch
from dataclasses import dataclass, field
from clip.constant import DEVICE
import torch.nn as nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from clip.constant import IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH


@dataclass
class ViTConfig:
    device: torch.device = field(default=DEVICE)
    n_layer: int = field(default=12, metadata={"help": "number of layers"})
    n_head: int = field(default=8, metadata={"help": "number of heads"})
    n_embd: int = field(default=512, metadata={"help": "embedding dimension"})
    mlp_size: int = field(default=2048, metadata={"help": "size of mlp"})
    patch_size: int = field(default=16, metadata={"help": "size of each image patch"})
    out_dim: int = field(default=768, metadata={"help": "output dimension"})
    img_channel: int = field(
        default=IMAGE_CHANNEL, metadata={"help": "input image channel"}
    )
    img_height: int = field(
        default=IMAGE_HEIGHT, metadata={"help": "input image height"}
    )
    img_width: int = field(default=IMAGE_WIDTH, metadata={"help": "input image width"})


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.c, self.h, self.w = config.img_channel, config.img_height, config.img_width
        self.psize = config.patch_size
        assert (
            self.h % self.psize == 0 and self.w % self.psize == 0
        ), "Image dimensions must be divisible by the patch size. Please check the configuration."
        self.n_patch = (self.h * self.w) // (self.psize**2)
        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.psize, p2=self.psize
        )
        self.pos_embd = nn.Embedding(self.n_patch + 1, self.config.n_embd)
        self.flatten = nn.Linear(
            self.c * self.psize**2, self.config.n_embd, bias=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, self.config.n_embd))

        self.ln = nn.LayerNorm(self.config.n_embd)
        self.transformer = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.mlp_head = nn.Linear(config.n_embd, config.out_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # chunk image to patches --> (B, npatch, nc * psize**2)
        x = self.rearrange(x)

        # flatten 2D image --> (B, npatch, nembd)
        x = self.flatten(x)

        # attach cls_token --> (B, npatch + 1, nembd)
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embd(
            torch.arange(x.shape[1], dtype=torch.long, device=self.config.device)
        )
        x = self.ln(x)
        for block in self.transformer:
            x = block(x)

        # getting class token --> (B, nembd)
        x = x[:, 0]

        # getting encoded output --> (B, out_dim)
        x = self.mlp_head(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = AttentionBlock(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.mlp_size),
            nn.GELU(),
            nn.Linear(config.mlp_size, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0

        self.w_qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        self.w_out = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x):
        B, L, D = x.shape  # batch size, sequence length, embedding dimension
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(
            1, 2
        )  # (B, nh, L, hs)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.w_out(out)
        return out
