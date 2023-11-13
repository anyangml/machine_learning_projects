import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """
        Parameters:
        ----------
        embedding_dim   : int
            dimension of the embedding, 512 in the original paper.
        num_heads       : int
            number of attention heads, 8 in the original paper.
        """
        super().__init__()

        msg = f"Embedding dimension '{embedding_dim}' is not divisible by number of attention heads '{num_heads}'."
        assert embedding_dim % num_heads == 0, msg

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qkv_dim = embedding_dim // num_heads

        # stack all attention heads together
        # the weight of `qkv_stacked_proj` has shape (num_heads * 3 * qkv_dim, embedding_dim)
        self.qkv_stacked_proj = nn.Linear(
            embedding_dim, 3 * num_heads * self.qkv_dim, bias=False
        )
        # the linear transformation Wo in the original paper
        self.output_proj = nn.Linear(
            num_heads * self.qkv_dim, embedding_dim, bias=False
        )
        self._initialize_para()

    def _initialize_para(self) -> None:
        nn.init.xavier_normal_(self.qkv_stacked_proj.weight)
        nn.init.xavier_normal_(self.output_proj.weight)

    def _self_attention_projection(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Perform matmul on X with stacked weight matrix to get query, key, value.
        Slice the resulting matrix to separate query, key, value.

        Parameters:
        ----------
        X   :   torch.Tensor
            the input embeddings of shape (batch_size, sequence_length, embedding_dim).

        Return
        ------
        q   :   torch.Tensor
            query matrix (Q) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        k   :   torch.Tensor
            key matrix (K) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        v   :   torch.Tensor
            value matrix (V) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        """

        batch_size, sequence_length, _ = X.shape
        qkv = self.qkv_stacked_proj(X)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(
        self,
        encoder_hidden_states: torch.Tensor,
        decoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Cross attention using decoder Query on encoder Key, Value.

        Parameters:
        ----------
        encoder_hidden_states   :   torch.Tensor
            the output of encoder block
        decoder_hidden_states   :   torch.Tensor
            the output of decoder block from the previous output token.

        """
        batch_size, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        batch_size, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

        # Split weight matrix
        w_q, w_kv = self.qkv_stacked_proj.weight.split([hidden_dim, 2 * hidden_dim])

        # Project encoder_hidden_states into k's, and v's
        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv)
            .reshape(batch_size, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        # Project decoder hidden states into q's
        q = F.linear(input=decoder_hidden_states, weight=w_q).reshape(
            batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        return q, k, v

    def _scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Perform scaled dot product attention.

        Parameters:
        ----------
        q   :   torch.Tensor
            query matrix (Q) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        k   :   torch.Tensor
            key matrix (K) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        v   :   torch.Tensor
            value matrix (V) of shape (batch_size, sequence_length, num_heads, qkv_dim).
        padding_mask    :   Optional[torch.BoolTensor]
            mask for padding tokens.
        future_mask    :   Optional[torch.BoolTensor]
            mask used in decoder for future tokens.

        Return
        ------
        values   :   torch.Tensor
            weighted sum of value vectors for each input token using attention scores.
        """

        # swap dimensions to (batch_size, num_heads, sequence_length, qkv_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # compute un-standardized attention weights of shape (batch_size, num_heads, seq_length, seq_length)
        attn_weights = torch.matmul(
            q,
            torch.transpose(k, -2, -1),
        )

        # scale attention weights by constant to create less spiky softmax distribution
        attn_weights = attn_weights / math.sqrt(q.size()[-1])

        # Apply attention mask (for pad tokens and future-masking in cross-attention)
        if padding_mask is not None or future_mask is not None:
            attn_weights = self.apply_mask(attn_weights, padding_mask, future_mask)

        # calculate probability distribution
        attention = F.softmax(attn_weights, dim=-1)

        # new contextualized representation
        # (batch_size, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)
        return values

    @staticmethod
    def apply_mask(
        attn_weights: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Reshape masks to fit the shape of the attention weights and set all indices with "False" to -inf

        Parameters:
        ----------
        attn_weights    :   torch.Tensor
            standardized attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        padding_mask    :   Optional[torch.BoolTensor]
            padding token flag of shape (batch_size, seq_length).
        future_mask    :   Optional[torch.BoolTensor]
            decoder self-attention to avoid any token i attending to a token >i
        Returns:
        ------
        masked_attn_weights     :    torch.Tensor
            masked attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        """
        if padding_mask is not None:
            masked_attn_weights = attn_weights.masked_fill(
                padding_mask[:, None, None, :] == 0, float("-inf")
            )
        if future_mask is not None:
            masked_attn_weights = attn_weights.masked_fill(
                future_mask == 0, float("-inf")
            )
        return masked_attn_weights

    def forward(
        self,
        X: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Perform scaled dot product attention.

        Parameters:
        ----------
        X   :   torch.Tensor
            the input embeddings of shape (batch_size, sequence_length, embedding_dim).
        encoder_hidden_states   :   Optional[torch.Tensor]
            the output of encoder blocks.
        padding_mask    :   Optional[torch.BoolTensor]
            mask for padding tokens.
        future_mask    :   Optional[torch.BoolTensor]
            mask used in decoder for future tokens.
        Return
        ------
        output   :   torch.Tensor
            the multi-head attention output after linear transformation, same shape as input X.
        """

        batch_size, sequence_length, embedding_dim = X.size()

        if encoder_hidden_states is None:
            # encoder block
            q, k, v = self._self_attention_projection(X)
        else:
            # decoder block
            q, k, v = self._cross_attention_projection(encoder_hidden_states, X)

        # Compute (contextualized) value vector for each "head"
        values = self._scaled_dot_product(q, k, v, padding_mask, future_mask)

        # Concatenate contextualized value vectors from all heads
        values = values.reshape(batch_size, sequence_length, embedding_dim)

        # Linearly transform the concatenation of all heads' value vectors (8*64=512) to the original hidden dim (512)
        output = self.output_proj(values)
        return output
