import numpy as np
import pytest

from ..conv_animation.convolution import convImage
from ..conv_animation.image_processing import readImage


@pytest.mark.parametrize("img_path", ["../data/img.png"])
def test_img(img_path: str) -> None:
    img = readImage(img_path)
    assert type(img).__module__ == np.__name__
    assert len(img.shape) == 3 or len(img.shape) == 2


def test_conv(img_path: str) -> None:
    img, conved_img, conv = convImage(img_path)
    weight = conv.weight.detach().numpy()
    assert type(img).__module__ == np.__name__
    assert len(img.shape) >= 3
    assert type(conved_img).__module__ == np.__name__
    assert len(conved_img.shape) == 4
    assert img.shape[0] == weight.shape[1]
    assert conved_img.shape[1] == weight.shape[0]
