from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from convolution import convImage
from matplotlib.animation import FuncAnimation


def showImage(img_path: str) -> None:
    """show duble images

    Args:
        img_path (str): image path
    """
    point = 47
    idx = 0
    img, conved_img, conv = convImage(img_path)
    if img.shape[0] == 1:
        img = img.squeeze()
    else:
        img = img.transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 2, tight_layout=True)
    weight = conv.weight.detach().numpy()[idx][0]
    weight = (weight - weight.min()) / (weight.max() - weight.min())
    conved_img = conved_img[0][idx]

    h, w = weight.shape
    dx = w // 2 + point % (img.shape[1] - 2 * (w // 2))
    dy = h // 2 + point // (img.shape[0] - 2 * (h // 2))
    img[dy - h // 2 : dy - h // 2 + h, dx - w // 2 : dx - w // 2 + w] = weight

    axes[0].imshow(img, cmap="gray")
    r = patches.Rectangle(
        xy=(dx - w // 2 - 0.5, dy - h // 2 - 0.6), width=w, height=h, ec="red", fill=False
    )
    axes[0].add_patch(r)
    dx = point % (conved_img.shape[1])
    dy = point // (conved_img.shape[0])
    axes[1].imshow(conved_img, cmap="gray")
    r = patches.Rectangle(xy=(dx - 0.5, dy - 0.6), width=1, height=1, ec="red", fill=False)
    axes[1].add_patch(r)
    plt.savefig("./image.png")
    plt.show()


def makeAnimation(img_path: str) -> None:
    """make animation

    Args:
        img_path (str): image path
    """
    idx = 0
    img, conved_img, conv = convImage(img_path)
    if img.shape[0] == 1:
        img = img.squeeze()
    else:
        img = img.transpose(1, 2, 0)
    weight = conv.weight.detach().numpy()[idx][0]
    weight = (weight - weight.min()) / (weight.max() - weight.min())
    h, w = weight.shape

    fig, axes = plt.subplots(1, 2, tight_layout=True)

    def update(
        frame: int,
        img: Any,
        weight: Any,
        conved_img: Any,
        idx: int = 0,
    ) -> None:
        """upadate visualization for animation

        Args:
            frame (int): the number of frames for animation
            img (Any): images
            weight (Any): Conv weights
            conved_img (Any): conved image
            idx (int, optional): index of channel. Defaults to 0.
        """
        img_copy = img.copy()
        conved_img_copy = conved_img[0][idx].copy()
        mask = np.zeros(conved_img_copy.shape).flatten()
        mask[frame + 1 :] = 1
        mask = mask.reshape(conved_img_copy.shape)
        conved_img_copy[mask == 1] = conved_img_copy.max()  # 白色にするため最大値

        dx = w // 2 + frame % (img.shape[1] - 2 * (w // 2))
        dy = h // 2 + frame // (img.shape[0] - 2 * (h // 2))
        img_copy[dy - h // 2 : dy - h // 2 + h, dx - w // 2 : dx - w // 2 + w] = weight

        axes[0].cla()
        axes[1].cla()
        axes[0].imshow(img_copy, cmap="gray")
        r = patches.Rectangle(
            xy=(dx - w // 2 - 0.5, dy - h // 2 - 0.6), width=w, height=h, ec="red", fill=False
        )
        axes[0].add_patch(r)

        dx = frame % (conved_img_copy.shape[1])
        dy = frame // (conved_img_copy.shape[0])
        axes[1].imshow(
            conved_img_copy,
            cmap="gray",
            vmin=conved_img.min(),
            vmax=conved_img.max(),
        )
        r = patches.Rectangle(xy=(dx - 0.5, dy - 0.6), width=1, height=1, ec="red", fill=False)
        axes[1].add_patch(r)

    def init() -> None:
        """init pass"""
        pass

    anim = FuncAnimation(
        fig,
        update,
        frames=range((img.shape[0] - 2 * (w // 2)) * (img.shape[1] - 2 * (h // 2))),
        interval=50,
        fargs=(img, weight, conved_img, idx),
        init_func=init,
    )

    anim.save("./animation.gif")
    plt.close()


def main() -> None:
    img_path = "./data/img.png"
    showImage(img_path)
    makeAnimation(img_path)


if __name__ == "__main__":
    main()
