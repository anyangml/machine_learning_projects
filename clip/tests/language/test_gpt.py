from clip.languange.gpt import GPT
from clip.languange.gpt import GPTConfig
import torch

def test_GPT_forward_shape():
    config = GPTConfig(device=torch.device("cpu"))
    gpt = GPT(config)
    dummy_txts = torch.randint(0, config.vocab_size, (2, 1024))
    ecoded = gpt(dummy_txts)
    assert ecoded.shape == (2, 1024, config.vocab_size)
    