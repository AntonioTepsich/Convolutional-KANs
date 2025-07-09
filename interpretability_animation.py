from manim import *
import numpy as np
import torch
import os
from architectures_28x28.KANConvs_MLP import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_model = KANC_MLP(grid_size=20)
random_model.to(device)
model = torch.load(
    os.path.join("models/FashionMNIST_torchlike/KANC MLP (Medium) (gs = 20).pt"),
    map_location=torch.device(device),
)


class ConvolutionAnimation(Scene):
    def create_grid(self, pixels):
        """Create a 10x10 grid based on the pixel values (0 or 255)."""
        grid = VGroup()
        # Normalize the convolved values to range 0 to 1 for grayscale mapping
        normalized_pixels = pixels / 255.0
        for i, row in enumerate(normalized_pixels):
            for j, value in enumerate(row):
                color = grayscale_to_color(value)
                square = Square(side_length=0.3, fill_opacity=1, color=color).move_to(
                    np.array([j * 0.3, -i * 0.3, 0])
                )
                grid.add(square)
        grid.move_to(3 * LEFT)
        return grid

    def convolution_result(self, pixels, kanconv_layer):
        """Compute the convolution of the pixel matrix with the given kernel."""
        pixels_tensor = (
            torch.tensor(pixels, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        convolved_tensor = kanconv_layer.forward(pixels_tensor / 255)
        convolved_tensor = torch.nn.ReLU()(convolved_tensor)

        convolved = convolved_tensor.squeeze().detach().cpu().numpy()
        convolved = convolved[3]  # Use the result of the first convolution in the layer
        normalized_convolved = convolved

        # kernel = [[0,0,0],[-1,0,1],[0,0,0]]
        # kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # convolved_tensor = F.conv2d(pixels_tensor, kernel_tensor, padding=(0,0))
        # normalized_convolved = torch.nn.ReLU()(convolved_tensor)/ 255
        # normalized_convolved = convolved_tensor.squeeze().detach().cpu().numpy() #not to index if done with torch

        # Create the result grid
        result_grid = VGroup()
        for i, row in enumerate(normalized_convolved):
            for j, value in enumerate(row):
                color = grayscale_to_color(value)
                square = Square(side_length=0.3, fill_opacity=1, color=color).move_to(
                    np.array([j * 0.3, -i * 0.3, 0])
                )
                result_grid.add(square)
        result_grid.move_to(3 * RIGHT)
        return result_grid

    def construct(self):
        title = Text(
            "KAN Convolution Results Animation", font_size=36, color=YELLOW
        ).to_edge(UP)

        # Define common patterns for the first convolutional layer
        patterns = [
            # np.zeros((10, 10), dtype=int),                # All pixels off
            # np.ones((10, 10), dtype=int) * 255,           # All pixels on
            # np.eye(10, dtype=int) * 255,                  # Diagonal line
            # np.fliplr(np.eye(10, dtype=int)) * 255,       # Other diagonal
            # np.array([[255] * 10 if i % 2 == 0 else [0] * 10 for i in range(10)]), # Horizontal stripes
            # np.array([[255 if (i + j) % 2 == 0 else 0 for j in range(10)] for i in range(10)]),  # Checkerboard
            # np.array([[255 if j < 5 else 0 for j in range(10)] for i in range(10)]),  # Vertical half-filled
            # np.array([[255 if i < 5 else 0 for j in range(10)] for i in range(10)]),  # Horizontal half-filled
            # np.pad(np.ones((6, 6)) * 255, pad_width=2, mode='constant'),  # Centered square
            # np.array([[i * 25 for j in range(10)] for i in range(10)]),   # Vertical gradient
            # np.array([[j * 25 for j in range(10)] for i in range(10)]),   # Horizontal gradient
            # np.array([[255 if j < 5 and i < 5 else 0 for j in range(10)] for i in range(10)]),  # Top-left quarter filled
            np.array(
                [
                    [255 if (i - 5) ** 2 + (j - 5) ** 2 <= 9 else 0 for j in range(10)]
                    for i in range(10)
                ]
            ),  # Circle
            np.array(
                [
                    [255 if (i == 5 or j == 5) else 0 for j in range(10)]
                    for i in range(10)
                ]
            ),  # Cross
            # New patterns
            np.array(
                [[255 if i + j < 10 else 0 for j in range(10)] for i in range(10)]
            ),  # Diagonal half-white
            np.array(
                [[255 if i > j else 0 for j in range(10)] for i in range(10)]
            ),  # Below diagonal
            np.array(
                [[255 if i < 5 and j < 5 else 0 for j in range(10)] for i in range(10)]
            ),  # Top-left quarter
            np.array(
                [[255 if i < j else 0 for j in range(10)] for i in range(10)]
            ),  # Below diagonal
            np.array(
                [[125 if i < j else 0 for j in range(10)] for i in range(10)]
            ),  # Below diagonal 2
            np.array(
                [[30 if i < j else 0 for j in range(10)] for i in range(10)]
            ),  # Below diagonal 3
            np.array(
                [[255 if i + j > 9 else 0 for j in range(10)] for i in range(10)]
            ),  # Above diagonal
        ]
        # Normalize the gradient patterns to ensure visibility
        patterns[-2] = (patterns[-2] / 255) * 255  # Ensure values are 0-255
        patterns[-1] = (patterns[-1] / 255) * 255  # Ensure values are 0-255
        # Display the title at the beginning
        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeOut(title))

        # Display each pattern and its convolution result
        for i, pixels in enumerate(patterns):
            # Create original grid
            original_grid = self.create_grid(pixels)
            # Convolution result grid
            result_grid = self.convolution_result(pixels, model.conv1)

            # Add labels and borders
            input_label = Text("Input", font_size=24).next_to(original_grid, UP)
            output_label = Text("Output", font_size=24).next_to(result_grid, UP)
            input_border = SurroundingRectangle(original_grid, color=BLUE)
            output_border = SurroundingRectangle(result_grid, color=GREEN)

            # Animate original and convolved grids with labels and borders
            self.play(
                FadeIn(original_grid),
                FadeIn(result_grid),
                FadeIn(input_label),
                FadeIn(output_label),
                Create(input_border),
                Create(output_border),
            )
            self.wait(2)
            self.play(
                FadeOut(original_grid),
                FadeOut(result_grid),
                FadeOut(input_label),
                FadeOut(output_label),
                FadeOut(input_border),
                FadeOut(output_border),
            )


def grayscale_to_color(value):
    """Helper function to convert a grayscale value to a color in Manim."""
    return interpolate_color(BLACK, WHITE, value)
