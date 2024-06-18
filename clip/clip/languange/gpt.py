import torch
from dataclasses import dataclass, field
from ..constant import DEVICE
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    device: torch.device = field(default=DEVICE)
    vocab_size: int = field(
        default=32768, metadata={"help": "define size of vocabulary"}
    ) # 2**15
    seq_len: int = field(default=1024, metadata={"help": "sequence length"})
    n_layer: int = field(default=12, metadata={"help": "number of layers"})
    n_head: int = field(default=8, metadata={"help": "number of heads"})
    n_embd: int = field(default=768, metadata={"help": "embedding dimension"})


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.seq_len, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
        self.ff = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.token_embd.weight = self.ff.weight


    def forward(self, x):
        seq_len = x.size(1)

        # (B, L) --> (B, L, D)
        token_embd = self.token_embd(x)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.config.device).unsqueeze(0)
        pos_embd = self.pos_embd(pos)
        x = token_embd + pos_embd
        
        
        for block in self.transformer:
            x = block(x)
        x = self.ln(x)

        # (B, L, D) --> (B, L, V)
        x = self.ff(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = AttentionBlock(config)
        self.ff = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0

        self.w_qkv = nn.Linear(self.n_embd, self.n_embd * 3)
        self.w_out = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x):
        B, L, D = x.shape # batch size, sequence length, embedding dimension
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(1, 2) # (B, nh, L, hs)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.w_out(out)
        return out