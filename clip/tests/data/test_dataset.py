from clip.data.dataset import CLIPDataset
import pytest
import torch
from clip.constant import IMAGE_HEIGHT, IMAGE_WIDTH


@pytest.fixture
def dataset(mock_data):
    return CLIPDataset(data_dir=mock_data)


def test_len(dataset):
    assert len(dataset) == 2


def test_getitem(dataset):
    img, txt = dataset[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(txt, list)
    assert img.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert txt[-1] == 50256
