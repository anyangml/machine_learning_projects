import torch
from torch import nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__() 
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, X):
        X = self.token_embed(X) + self.pos_embed[:,:X.shape[1],:]
        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_dim = int(embed_dim // num_heads)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, X, mask=None):
        batch_size, seq_len, _ = X.shape
        qkv = self.qkv_proj(X)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1) #  (batch_size, sequence_len, num_heads, embd_dim)

        attn = torch.einsum('bqnd,bknd->bnqk', [q, k])
        attn = attn / ((q.size()[-1]) ** (1 / 2))
        
        if mask is not None:
            attn = attn.masked_fill(mask[:seq_len,:seq_len] == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bnqk,bknd->bqnd', [attn, v]).reshape(batch_size,seq_len, self.num_heads * self.qkv_dim)
        out = self.out_proj(out)
        return out

class GPTBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout_p, ff_ratio):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(ff_ratio * embed_dim, embed_dim)
        )

    def forward(self, X, mask=None):
        out = self.attn(X, mask)
        out = self.norm1(X + self.dropout(out))
        out = self.ff(out)
        out = self.norm2(out + self.dropout(out))
        return out

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout_p, ff_ratio, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.embed = GPTEmbedding(vocab_size, embed_dim, max_len)
        self.block = GPTBlock(embed_dim, num_heads, dropout_p, ff_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Parameter, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, X):
        out = self.embed(X)
        mask = self._generate_mask(X)
        for _ in range(self.num_layers):
            out = self.block(out, mask)
        out = self.norm(out)
        out = self.ff(out)
        out = self.softmax(out)
        return out
    
    def _generate_mask(self, X):
        mask = torch.ones(X.shape[1], X.shape[1])
        mask = torch.tril(mask)
        return mask