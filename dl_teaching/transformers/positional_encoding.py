from torch import nn
import torch


class PositionalEncoding(nn.Module):
    """Add positional encoding to the embedded input data."""

    def __init__(self, embedding_dim: int, max_seq_len: int = 1000) -> None:
        """
        Parameters:
        ----------
        embedding_dim   : int
            dimension of the embedding, 512 in the original paper.
        max_seq_len     : int
            the maximum number of tokens allowed in a single sequence.
        """
        super().__init__()

        self.P = torch.zeros((1, max_seq_len, embedding_dim))
        X = torch.arange(max_seq_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim,
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        X   :    torch.Tensor
            the input in shape (batch_size, sequence_length, embedding_dim).

        Returns:
        -------
        X   :   torch.Tensor
            the position encoded input.
        """
        X = X + self.P[:, : X.shape[1], :]
        return X
