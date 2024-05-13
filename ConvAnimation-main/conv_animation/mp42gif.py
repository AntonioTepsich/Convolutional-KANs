from moviepy.editor import VideoFileClip


def main() -> None:
    input_path = "media/ConvAnim.mp4"
    output_path = "media/ConvAnim.gif"
    width = 600
    fps = 10

    clip = VideoFileClip(input_path)
    clip = clip.resize(width=width)
    clip.write_gif(output_path, fps=fps)
    clip.close()


if __name__ == "__main__":
    main()
