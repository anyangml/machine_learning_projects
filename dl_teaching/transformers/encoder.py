import torch
from torch import nn
import math
from typing import Optional

from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """The encoder block"""

    def __init__(
        self, embedding_dim: int, ff_dim: int, num_heads: int, dropout_p: float
    ) -> None:
        """
        Parameters:
        ----------
        embedding_dim   : int
            dimension of the embedding, 512 in the original paper.
        ff_dim       : int
            dimension of the position-wise feed-forward networks, 2048 in the original paper.
        num_heads       : int
            number of attention heads, 8 in the original paper.
        dropout_p   :   float
            residual dropout, 0.1 in the original paper.
        """
        super().__init__()
        self.self_mha = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, X: torch.FloatTensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Performs one encoder *block* forward pass given the previous block's output and an optional attention mask.

        Parameters:
        ----------
        X   :  torch.FloatTensor
            tensor containing the output of the previous encoder block of shape (batch_size, seq_length, embedding_dim).
        padding_mask    :   Optional[torch.BoolTensor]
            an attention mask to ignore pad-tokens in the input of shape(batch_size, seq_length).

        Returns:
        -------
        X   :  torch.FloatTensor
            updated intermediate encoder (contextualized) token embeddings of the same shape.
        """
        output = self.dropout1(self.self_mha.forward(X, padding_mask=padding_mask))
        X = self.layer_norm1(X + output)

        output = self.dropout2(self.feed_forward(X))
        X = self.layer_norm2(X + output)
        return X


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        embedding_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
    ) -> None:
        """
        Parameters:
        ----------
        embedding   :   torch.nn.Embedding
            the embedding object that performs the embedding transformation (vocab_size, embedding_dim).
        embedding_dim   : int
            dimension of the embedding, 512 in the original paper.
        ff_dim       : int
            dimension of the position-wise feed-forward networks, 2048 in the original paper.
        num_heads       : int
            number of attention heads, 8 in the original paper.
        num_layers       : int
            number of encoder blocks, 6 in the original paper.
        dropout_p   :   float
            residual dropout, 0.1 in the original paper.
        """

        super().__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=5000)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embedding_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self, input_ids: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Performs one encoder forward pass given input token ids and an optional attention mask.

        Parameters:
        ----------
        input_ids   :   torch.Tensor
            the raw input token ids of shape (batch_size, seq_lenght).
        padding_mask    :   Optional[torch.BoolTensor]
            an attention mask to ignore pad-tokens in the input of shape (batch_size, seq_length).

        Returns:
        -------
        X   :   torch.Tensor
            the encoder's final (contextualized) token embeddings of shape (batch_size, seq_length, embedding_dim),
            used in decoder cross attention.
        """
        # the resulting embedding is scaled by sqrt(embedding_dim) before adding positional encoding.
        X = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        X = self.positional_encoding(X)
        X = self.dropout(X)
        for encoder_block in self.encoder_blocks:
            X = encoder_block.forward(X, padding_mask=padding_mask)
        return X
