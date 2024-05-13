from typing import Any, Tuple

import torch.nn as nn
from image_processing import readImage
from torchvision import transforms as transforms


def convImage(img_path: str) -> Tuple[Any, Any, Any]:
    """Convolution of an image

    Args:
        img_path (str): image path

    Returns:
        (Any): numpy img
        (Any): conved numpy img
        (Any): conv weights
    """
    img = readImage(img_path)
    conv = nn.Conv2d(1, 16, 3)
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_conv = conv(img_tensor)
    print("shape",img_tensor.numpy().shape)
    return img_tensor.numpy()[0], img_conv.detach().numpy(), conv


def main() -> None:
    img_path = "./data/img.png"
    convImage(img_path)


if __name__ == "__main__":
    main()
