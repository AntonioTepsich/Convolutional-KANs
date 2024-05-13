from image_processing import writeMNIST


def main() -> None:
    data_path = "./data"
    idx = 100
    img_path = "img.png"
    writeMNIST(data_path, img_path, idx)


if __name__ == "__main__":
    main()
