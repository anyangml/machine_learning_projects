from clip.languange.gpt import GPT
from clip.languange.gpt import GPTConfig
import torch


def test_GPT_forward_shape():
    config = GPTConfig(device=torch.device("cpu"))
    gpt = GPT(config)
    dummy_txts = torch.randint(0, config.vocab_size, (2, 1024))
    # add EOS token
    dummy_txts[0, 152] = 50256
    dummy_txts[1, 928] = 50256

    encoded = gpt(dummy_txts)
    assert encoded.shape == (2, config.vocab_size)
