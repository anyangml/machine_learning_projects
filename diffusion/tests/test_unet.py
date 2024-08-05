from diffusion import UNet, DownSampleBlock, UNetWithTime
import torch


# def test_downsample_block_shape():
#     initial_channels = 64
#     initial_size = 568
#     x = torch.randn(2, initial_channels, initial_size, initial_size)

#     for _ in range(4):
#         x = DownSampleBlock(initial_channels, initial_channels * 2)(x)
#         assert x.shape == (
#             2,
#             initial_channels * 2,
#             initial_size // 2 - 4,
#             initial_size // 2 - 4,
#         )
#         initial_channels *= 2
#         initial_size = initial_size // 2 - 4


# def test_unet_forward_pass_shape():
#     model = UNet()
#     x = torch.randn(2, 1, 572, 572)
#     y = model(x)
#     assert y.shape == (2, 2, 388, 388)


def test_unet_forward_pass_with_time():
    model = UNetWithTime()
    x = torch.randn(2, 1, 512, 512)
    y = model(x, t=torch.randint(0, 5, (2,)).to(torch.float32))
    assert y.shape == x.shape
