from typing import Optional
import math
import numpy as np
import torch
from torch import nn

from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, ff_dim: int, num_heads: int, dropout_p: float
    ) -> None:
        super().__init__()

        self.cross_mha = MultiHeadAttention(embedding_dim, num_heads)
        self.self_mha = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        X: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        """
        Performs one decoder *block* forward pass given final encoder hidden states, the previous block's output, and
        attention masks.

        Parameters:
        ---------
        X   :   torch.Tensor
            previous decoder block's output of shape (batch_size, seq_length, embedding_dim).
        encoder_hidden_states   :   torch.FloatTensor
            encoder's final (contextualized) token embeddings of shape (batch_size, seq_length, embedding_dim).
        padding_mask    :   Optional[torch.BoolTensor]
            an attention mask to ignore pad-tokens in the input of shape(batch_size, seq_length).
        future_mask    :   Optional[torch.BoolTensor]
            an attention mask to ignore future-tokens in the target of shape(seq_length, seq_length).

        Returns:
        -------
        X   :   torch.Tensor
            updated contextualized token embeddings of shape (batch_size, seq_length, embedding_dim).
        """

        # self attention (with future masking during training)
        output = self.dropout1(self.self_mha.forward(X, future_mask=future_mask))
        X = self.layer_norm1(X + output)

        # encoder-decoder cross attention
        output = self.dropout2(
            self.cross_mha.forward(
                X,
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
            )
        )
        X = self.layer_norm2(X + output)

        # feed forward layers
        output = self.dropout3(self.feed_forward(X))
        X = self.layer_norm3(X + output)
        return X


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        embedding_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        dropout_p: float,
    ) -> None:
        """
        Parameters:
        ----------
        embedding   :   torch.nn.Embedding
            the embedding object that performs the embedding transformation.
        embedding_dim   : int
            dimension of the embedding, 512 in the original paper.
        ff_dim       : int
            dimension of the position-wise feed-forward networks, 2048 in the original paper.
        num_heads       : int
            number of attention heads, 8 in the original paper.
        num_layers       : int
            number of decoder blocks, 6 in the original paper.
        vocab_size  :   int
            target vocabulary size.
        dropout_p   :   float
            residual dropout, 0.1 in the original paper.
        """
        super().__init__()

        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embedding_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)

    def _reset_parameters(self):
        """perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        input_tokens: torch.IntTensor,
        encoder_hidden_states: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Performs one decoder forward pass given encoder hidden states, the decoder input tokens and attention masks.

        Parameters:
        ----------
        input_tokens   :   torch.IntTensor
            the raw input token ids of shape (batch_size, seq_lenght).
        encoder_hidden_states   :   torch.Tensor
            the output of encoder blocks.
        padding_mask    :   Optional[torch.BoolTensor]
            an attention mask to ignore pad-tokens in the input of shape (batch_size, seq_length).
        future_mask    :   Optional[torch.BoolTensor]
            mask used in decoder for future tokens.


        Returns:
        -------
        logits   :   torch.Tensor
            unnormalized logits over the vocabulary for every token in the batch of shape (batch_size, seq_length, vocab_size).
        """
        # (batch_size, sequence_length, embedding_dim)
        # since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        x = self.embedding(input_tokens) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, padding_mask, future_mask)

        # (batch_size, sequence_length, vocab_size)
        logits = self.output_layer(x)
        return logits
