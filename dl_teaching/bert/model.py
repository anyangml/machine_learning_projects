import torch
from torch import nn
from typing import Tuple


class BertEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 30000,
        max_len: int = 500,
        embd_dim: int = 768,
        num_seg: int = 2,
    ) -> None:
        """
        Learnable embedding that convert encoded input tensor and segment encoding
        into embedding tensor.

        Parameters:
        ----------
        vocab_size: int
            The size of vocabulary, 30000 in the original paper.
        max_len:    int
            The maximum sequence length.
        embd_dim:   int
            The embedding dimension, 768 in the original paper.
        num_seg:    int
            The number of segments, 2 in the original paper (either 0 or 1).
        """
        super().__init__()

        # (size of the dictionary of embeddings, the size of each embedding tensor)
        self.word_embd = nn.Embedding(vocab_size, embd_dim)
        self.segment_embd = nn.Embedding(num_seg, embd_dim)
        self.position_embd = nn.Embedding(max_len, embd_dim)
        self.norm = nn.LayerNorm(embd_dim)

    def forward(self, X: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        X:  torch.Tensor
            The input encoding tensor of shape (batch_size, sequence_length).
        seg:    toch.Tensor
            The segment encoding of shape (batch_size, sequence_length).

        Returns:
        -------
        norm:   torch.Tensorp
            The combined embedding after layer normalization of shape (batch_size, sequence_length, embd_dim).
        """
        seq_len = X.shape[1]
        pos = torch.arange(seq_len).unsqueeze(0).expand_as(X)
        embd_sum = self.word_embd(X) + self.position_embd(pos) + self.segment_embd(seg)
        norm = self.norm(embd_sum)
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim: int = 768, num_heads: int = 12) -> None:
        """
        Parameters:
        ----------
        embd_dim:   int
            The embedding dimension, 768 in the original paper.
        num_heads:  int
            The number of attention heads, 12 in the original paper.
        """
        super().__init__()
        assert (
            embd_dim % num_heads == 0
        ), f"Embedding dimension '{embd_dim}' is not divisible by number of attention heads '{num_heads}'."
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.qkv_dim = int(embd_dim / num_heads)

        self.qkv_stacked_proj = nn.Linear(embd_dim, 3 * num_heads * self.qkv_dim, bias=False)

        self.fc = nn.Linear(self.qkv_dim * self.num_heads, self.embd_dim, bias=False)

    def _self_projection(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Perform matmul on X with stacked weight matrix to get query, key, value.
        Slice the resulting matrix to separate query, key, value.

        Parameters:
        ----------
        X:   torch.Tensor
            The input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Return
        ------
        q:   torch.Tensor
            Query matrix (Q) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        k:   torch.Tensor
            Key matrix (K) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        v:   torch.Tensor
            Value matrix (V) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        """
        batch_size, sequence_length, _ = X.shape
        qkv = self.qkv_stacked_proj(X)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Perform multi head attention on X with mask.

        Parameters:
        ----------
        X:   torch.Tensor
            The input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Returns:
        -------
        out:   torch.Tensor
            The output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        q, k, v = self._self_projection(X)
        batch_size, sequence_length, num_heads, qkv_dim = q.shape

        # (batch_size, sequence_length, num_heads, qkv_dim) to
        # (batch_size, num_heads, sequence_length, sequence_length)
        attn_weight = torch.einsum("bqhd,bkhd->bhqk", [q, k])
        attn_weight = attn_weight / ((q.size()[-1]) ** (1 / 2))

        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask[:,None,None,:] == 0, float("-inf"))

        attn_weight = torch.softmax(attn_weight, dim=-1)

        # (batch_size, num_heads, sequence_length, sequence_length) to
        # (batch_size, sequence_length, num_heads * qkv_dim)
        out = torch.einsum("bhqv,bvhd->bqhd", [attn_weight, v]).reshape(
            batch_size, sequence_length, num_heads * qkv_dim
        )
        out = self.fc(out)
        return out


class BertBlock(nn.Module):
    def __init__(
        self,
        embd_dim: int = 768,
        num_heads: int = 12,
        dropout_p: float = 0.1,
        ff_ratio: int = 4,
    ) -> None:
        """
        Parameters:
        ----------
        embd_dim:   int
            The embedding dimension, 768 in the original paper.
        num_heads:    int
            The number of attention heads, 12 in the original paper.
        dropout_p:  float
            The dropout ratio, 0.1 in the original paper.
        ff_ratio:   int
            The multiplier that determines the number of hidden nodes in the feed-forward layer.
        """
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embd_dim)
        self.layer_norm2 = nn.LayerNorm(embd_dim)
        self.attention = MultiHeadAttention(embd_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embd_dim, embd_dim * ff_ratio),
            nn.GELU(),
            nn.Linear(embd_dim * ff_ratio, embd_dim),
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Perform multi head attention on X with mask.

        Parameters:
        ----------
        X:   torch.Tensor
            The input embeddings from the previous BERT block of shape
            (batch_size, sequence_length, embedding_dim).

        Returns:
        -------
        out:   torch.Tensor
            The output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        attnt_block = self.attention(X, mask)
        residual_sum = self.dropout(self.layer_norm1(X + attnt_block))
        feed_forward = self.feed_forward(residual_sum)
        out = self.dropout(self.layer_norm2(feed_forward + residual_sum))
        return out


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embd_dim: int,
        num_seg: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
        ff_ratio: int,
    ) -> None:
        """
        Parameters:
        ----------
        vocab_size: int
            The size of vocabulary, 30000 in the original paper.
        max_len:    int
            The maximum sequence length.
        embd_dim:   int
            The embedding dimension, 768 in the original paper.
        num_seg:    int
            The number of segments, 2 in the original paper (either 0 or 1).
        num_heads:  int
            The number of attention heads, 12 in the original paper.
        num_layers: int
            The number of Bert blocks in the model, 12 in the original paper.
        dropout_p:  float
            The dropout ratio, 0.1 in the original paper.
        ff_ratio:   int
            The multiplier that determines the number of hidden nodes in the feed-forward layer.
        """
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, max_len, embd_dim, num_seg)
        self.bert_blocks = nn.ModuleList(
            [
                BertBlock(embd_dim,num_heads,dropout_p,ff_ratio)
                for _ in range(num_layers)
            ]
        )
        self.activ = nn.Tanh()
        self.norm = nn.LayerNorm(embd_dim)
        self.classifier = nn.Linear(embd_dim, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.fc = nn.Linear(embd_dim, embd_dim)
        embed_weight = self.embedding.word_embd.weight
        vocab_size, embd_dim = embed_weight.size()
        self.decoder = nn.Linear(embd_dim, vocab_size, bias=False)
        self.decoder.weight = embed_weight
        self.dropout = nn.Dropout(dropout_p)
        self.apply(self._init_weight)

    def _init_weight(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, X, seg, mask):
        X = self.dropout(self.embedding(X, seg))
        for block in self.bert_blocks:
            X = block.forward(X, mask)
        
        # isNext prediction.
        h_pooled = self.activ(self.fc(X[:, 0])) # [batch_size, embd_dim]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        # masked token prediction.
        logits_lm = self.softmax(self.decoder(X))
        return logits_lm, logits_clsf