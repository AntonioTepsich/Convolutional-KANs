from typing import Any

import numpy as np
import torchvision
from PIL import Image


def readImage(img_path: str) -> Any:
    """read an image from image_path

    Args:
        img_path (str): image path

    Returns:
        Any: numpy read image
    """
    img = Image.open(img_path)
    return np.array(img)


def writeImage(
    img: Any,
    img_path: str,
) -> None:
    """write an image to img_path

    Args:
        img (Any): numpy image
        img_path (str): image path
    """
    pil_img = Image.fromarray(img)
    pil_img.save(img_path)


def loadMNIST(data_path: str) -> Any:
    """load MNIST dataset

    Args:
        data_path (str): MNIST dataset path
    Returns:
        manist (Any): dataset
    """
    mnist = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
    )
    return mnist


def writeMNIST(
    data_path: str,
    img_path: str,
    idx: int = 0,
) -> None:
    """write an image of MNIST

    Args:
        data_path (str): MNIST dataset path
        img_path (str): image path to write
        idx (int, optional): image idx in dataset. Defaults to 0.
    """
    mnist = loadMNIST(data_path)
    img = np.array(mnist[idx][0])
    writeImage(img, img_path)
