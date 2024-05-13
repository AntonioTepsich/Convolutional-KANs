from typing import Any

import numpy as np
from convolution import convImage
from manim import (
    BLACK,
    DOWN,
    LEFT,
    RED,
    RESAMPLING_ALGORITHMS,
    RIGHT,
    UP,
    WHITE,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    ImageMobject,
    Line,
    MathTex,
    MovingCameraScene,
    Square,
    Tex,
    Uncreate,
    VGroup,
    config,
)


def img_unNorm(img: Any) -> Any:
    if (img.dtype == np.float32) | (img.dtype == np.float64):
        img = (img - img.min()) / (img.max() - img.min())
        img *= 255
        img = img.astype(np.uint8)
    return img


class ConvAnim(MovingCameraScene):
    def construct(self) -> None:
        img_path = "../data/img.png"
        img, conved_img, kernel = convImage(img_path)
        img = img_unNorm(img[0])
        conved_img = img_unNorm(conved_img)
        kernel = img_unNorm(kernel.weight.detach().numpy()[0][0])

        img_obj_size = 5
        kernel_size = kernel.shape[0]
        img_size = img.shape[0]
        kernel_obj_size = img_obj_size / img_size * kernel_size

        img_obj = ImageMobject(img)
        img_obj.height = img_obj_size
        img_obj.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        # image square
        img_square = Square(side_length=img_obj_size)
        img_square.set_stroke(WHITE, width=1, opacity=0.5)

        # image hlines
        img_hlines = VGroup()
        line = Line()
        for i in range(img_size - 1):
            img_hlines.add(line.copy())
            img_hlines[i].set_stroke(WHITE, width=1, opacity=0.5)
            img_hlines[i].set_length(img_obj_size)
            img_hlines[i].shift(
                img_obj.get_edge_center(UP) + img_obj_size / img_size * (i + 1) * DOWN
            )

        # image vlines
        img_vlines = VGroup()
        line.put_start_and_end_on(UP, DOWN)
        for i in range(img_size - 1):
            img_vlines.add(line.copy())
            img_vlines[i].set_stroke(WHITE, width=1, opacity=0.5)
            img_vlines[i].set_length(img_obj_size)
            img_vlines[i].shift(
                img_obj.get_edge_center(LEFT) + img_obj_size / img_size * (i + 1) * RIGHT
            )

        # image name
        img_name = Tex("Input Image", font_size=18)
        img_name.set_color(WHITE)
        img_name.shift(img_obj.get_edge_center(UP) + 0.1 * UP)

        # kernel
        kernel_obj = ImageMobject(kernel)
        kernel_obj.height = kernel_obj_size
        kernel_obj.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        kernel_obj.shift(
            img_obj.get_corner(UP + LEFT) + kernel_obj_size / 2 * RIGHT + kernel_obj_size * UP
        )

        # kernel name
        kernel_name = MathTex("Kernel", font_size=18)
        kernel_name.set_color(RED)
        kernel_name.shift(kernel_obj.get_edge_center(UP) + 0.1 * UP)

        # kernel square
        kernel_square = Square(side_length=kernel_obj_size)
        kernel_square.set_stroke(RED, width=2)
        kernel_square.shift(kernel_obj.get_corner(UP + LEFT) + kernel_obj_size / 2 * (DOWN + RIGHT))

        # kernel square on image
        kernel_square_on_img = Square(side_length=kernel_obj_size)
        kernel_square_on_img.set_stroke(RED, width=2)
        kernel_square_on_img.shift(
            img_obj.get_corner(UP + LEFT) + kernel_obj_size / 2 * (DOWN + RIGHT)
        )


        # kernel lines
        kernel_lines = VGroup()
        line = Line()
        for i in range(kernel_size - 1):
            kernel_lines.add(line.copy())
            kernel_lines[i].set_stroke(RED, width=2)
            kernel_lines[i].set_length(kernel_obj_size)
            kernel_lines[i].shift(
                kernel_obj.get_edge_center(UP) + img_obj_size / img_size * (i + 1) * DOWN
            )
        line.put_start_and_end_on(UP, DOWN)
        for i in range(kernel_size - 1):
            kernel_lines.add(line.copy())
            kernel_lines[kernel_size - 1 + i].set_stroke(RED, width=2)
            kernel_lines[kernel_size - 1 + i].set_length(kernel_obj_size)
            kernel_lines[kernel_size - 1 + i].shift(
                kernel_obj.get_edge_center(LEFT) + img_obj_size / img_size * (i + 1) * RIGHT
            )

        # kernel text
        kernel_text_on_img = VGroup()
        for i in range(kernel_size):
            for j in range(kernel_size):
                ind = f"{i+1},{j+1}"
                tex = f"\phi_{{{ind}}}"
                text = MathTex(tex, font_size=10)
                text.set_color(BLACK if kernel[i][j] > 128 else WHITE)
                text.shift(
                    kernel_square.get_corner(UP + LEFT) + img_obj_size / img_size / 2 * (DOWN + RIGHT)
                )
                text.shift(
                    img_obj_size / img_size * (j ) * RIGHT
                    + img_obj_size / img_size * (i) * DOWN
                )
                kernel_text_on_img.add(text)
        image_text_on_img = VGroup()
        for i in range(kernel_size):
            for j in range(kernel_size):
                ind = f"{j+1},{i+1}"
                tex = f"x_{{{ind}}}"
                text = MathTex(tex, font_size=10)
                #text.set_color(BLACK if kernel[i][j] > 128 else WHITE)
                text.shift(
                    kernel_square_on_img.get_corner(UP + LEFT) + img_obj_size / img_size / 2 * (DOWN + RIGHT)
                )
                text.shift(
                    img_obj_size / img_size * (i ) * RIGHT
                    + img_obj_size / img_size * (j) * DOWN
                )
                image_text_on_img.add(text)
        # kernel text on image
        kernel_text = VGroup()
        
        image_text = VGroup()

        # kernel connection
        kernel_connection1 = DashedLine(
            config.left_side, config.right_side, dash_length=0.9, dashed_ratio=0.5
        )
        kernel_connection1.set_stroke(RED, width=2)
        kernel_connection1.put_start_and_end_on(
            kernel_square.get_corner(DOWN + LEFT), kernel_square_on_img.get_corner(UP + LEFT)
        )
        kernel_connection2 = DashedLine(
            config.left_side, config.right_side, dash_length=0.9, dashed_ratio=0.5
        )
        kernel_connection2.set_stroke(RED, width=2)
        kernel_connection2.put_start_and_end_on(
            kernel_square.get_corner(DOWN + RIGHT), kernel_square_on_img.get_corner(UP + RIGHT)
        )

        # conved image name
        conved_name = Tex("Conved Image", font_size=18)
        conved_name.set_color(WHITE)

        # conved pixels
        conved_pixel_list = []
        for i in range(conved_img[0][0].shape[0]):
            _conved_pixel_list = []
            for j in range(conved_img[0][0].shape[1]):
                _conved_pixel_list.append(ImageMobject(np.uint8([[conved_img[0][0][i][j]]])))
                _conved_pixel_list[j].height = img_obj_size / img_size
                _conved_pixel_list[j].set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            conved_pixel_list.append(_conved_pixel_list)
        conved_pixel_list[0][0].shift(
            kernel_square.get_corner(RIGHT + DOWN) + kernel_obj_size * 2 * RIGHT
        )

        # conv equation
        tex = r" \phi_{11}( x_{11}) + \phi_{12} (x_{12})+ \phi_{13} (x_{13})+\\ \phi_{21} (x_{21}) + \phi_{22} (x_{22})+\phi_{23} (x_{23}) \\\ \phi_{31} (x_{31}) + \phi_{32} (x_{32})+\phi_{33} (x_{33})"
        conv_eq_text = MathTex(tex, font_size=12)
        conv_eq_text.shift(
            conved_pixel_list[0][0].get_edge_center(UP) + img_obj_size / img_size * 2 * UP
        )

        # kernel detail
        kernel_detail1 = Line(
            kernel_square.get_corner(RIGHT + UP), conved_pixel_list[0][0].get_corner(LEFT + UP)
        )
        kernel_detail1.set_stroke(RED, width=2)
        kernel_detail2 = Line(
            kernel_square_on_img.get_corner(RIGHT + DOWN),
            conved_pixel_list[0][0].get_corner(LEFT + DOWN),
        )
        kernel_detail2.set_stroke(RED, width=2)

        # show
        self.add(img_obj)
        self.play(Create(img_name))
        self.wait()
        self.play(Create(img_square), Create(img_hlines), Create(img_vlines))
        self.play(FadeIn(kernel_obj), FadeIn(kernel_square), FadeIn(kernel_lines))

        self.play(self.camera.frame.animate.scale(0.4).move_to(kernel_square_on_img))

        self.play(FadeIn(kernel_name))
        self.wait()

        self.play(Create(kernel_connection1), Create(kernel_connection2))
        self.play(FadeIn(kernel_square_on_img))

        self.play(FadeIn(kernel_text_on_img), FadeIn(kernel_text))
        self.play(FadeIn(image_text_on_img), FadeIn(image_text))

        self.add(conved_pixel_list[0][0])
        self.play(Create(kernel_detail1), Create(kernel_detail2), Create(conv_eq_text))
        self.wait(2)

        self.play(
            Uncreate(kernel_name),
            Uncreate(conv_eq_text),
            Uncreate(kernel_text),
            Uncreate(kernel_text_on_img),
            Uncreate(image_text_on_img),

        )
        self.play(Uncreate(kernel_detail1), Uncreate(kernel_detail2))
        self.play(
            FadeOut(kernel_square_on_img),
            Uncreate(kernel_connection1),
            Uncreate(kernel_connection2),
            kernel_square.animate.shift(
                kernel_square_on_img.get_corner(UP + LEFT) - kernel_square.get_corner(UP + LEFT)
            ),
            kernel_lines.animate.shift(
                kernel_square_on_img.get_corner(UP + LEFT) - kernel_lines.get_corner(UP + LEFT)
            ),
            kernel_obj.animate.shift(
                kernel_square_on_img.get_corner(UP + LEFT) - kernel_obj.get_corner(UP + LEFT)
            ),
        )

        self.play(self.camera.frame.animate.scale(1 / 0.4).move_to(img_obj).shift(1.5 * RIGHT))

        self.play(
            conved_pixel_list[0][0].animate.shift(
                img_square.get_corner(UP + RIGHT)
                - conved_pixel_list[0][0].get_corner(UP + LEFT)
                + kernel_obj_size * RIGHT
                + img_obj_size / img_size * DOWN
            ),
        )

        kernel_detail1.put_start_and_end_on(
            kernel_square.get_corner(RIGHT + UP), conved_pixel_list[0][0].get_corner(LEFT + UP)
        )
        kernel_detail2.put_start_and_end_on(
            kernel_square.get_corner(RIGHT + DOWN), conved_pixel_list[0][0].get_corner(LEFT + DOWN)
        )

        self.play(Create(kernel_detail1), Create(kernel_detail2))


        for i in range(conved_img[0][0].shape[0]):
            if i != 0:
                self.play(
                    kernel_square.animate.shift(
                        img_obj_size / img_size * j * LEFT + img_obj_size / img_size * DOWN
                    ),
                    kernel_lines.animate.shift(
                        img_obj_size / img_size * j * LEFT + img_obj_size / img_size * DOWN
                    ),
                    kernel_obj.animate.shift(
                        img_obj_size / img_size * j * LEFT + img_obj_size / img_size * DOWN
                    ),
                    kernel_detail1.animate.shift(
                        img_obj_size / img_size * j * LEFT + img_obj_size / img_size * DOWN
                    ),
                    kernel_detail2.animate.shift(
                        img_obj_size / img_size * j * LEFT + img_obj_size / img_size * DOWN
                    ),
                    run_time=0.1,
                )
                conved_pixel_list[i][0].shift(
                    conved_pixel_list[(i - 1)][0].get_center() + img_obj_size / img_size * DOWN
                )
                self.add(conved_pixel_list[i][0])
            for j in range(1, conved_img[0][0].shape[1]):
                self.play(
                    kernel_square.animate.shift(img_obj_size / img_size * RIGHT),
                    kernel_lines.animate.shift(img_obj_size / img_size * RIGHT),
                    kernel_obj.animate.shift(img_obj_size / img_size * RIGHT),
                    kernel_detail1.animate.shift(img_obj_size / img_size * RIGHT),
                    kernel_detail2.animate.shift(img_obj_size / img_size * RIGHT),
                    run_time=0.02,
                )
                conved_pixel_list[i][j].shift(
                    conved_pixel_list[i][j - 1].get_center() + img_obj_size / img_size * RIGHT
                )
                self.add(conved_pixel_list[i][j])

        conved_name.shift(
            conved_pixel_list[0][conved_img[0][0].shape[1] // 2].get_edge_center(UP) + 0.1 * UP
        )
        self.play(Create(conved_name))

        self.wait(4)
