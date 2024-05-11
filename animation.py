from manim import *

class ConvolutionAnimation(Scene):
    def construct(self):
        # Define input and kernel tensors
        input_tensor = [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]
        kernel = [[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]]

        # Create input tensor box
        input_box = self.create_tensor_box(input_tensor, color=BLUE)

        # Create kernel tensor box
        kernel_box = self.create_tensor_box(kernel, color=RED)
        kernel_box.next_to(input_box, RIGHT)

        # Create output tensor box
        output_tensor = self.convolve(input_tensor, kernel)
        output_box = self.create_tensor_box(output_tensor, color=GREEN)
        output_box.next_to(kernel_box, RIGHT)

        # Display input, kernel, and output tensors
        self.play(Write(input_box), Write(kernel_box), Write(output_box))
        self.wait()

    def create_tensor_box(self, tensor, color):
        box = VGroup()
        for i, row in enumerate(tensor):
            for j, val in enumerate(row):
                rect = Square(color=color, fill_opacity=0.7).scale(0.6)
                rect.move_to(0.8 * (j - len(row) / 2) * RIGHT + 0.8 * (len(tensor) / 2 - i) * UP)
                text = Text(f"{val}").scale(0.5)
                text.next_to(rect, UP)
                box.add(rect)
                box.add(text)
        return box

    def convolve(self, input_tensor, kernel):
        output_tensor = []
        kernel_size = len(kernel)
        for i in range(len(input_tensor) - kernel_size + 1):
            row = []
            for j in range(len(input_tensor[0]) - kernel_size + 1):
                output_val = 0
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        output_val += input_tensor[i + m][j + n] * kernel[m][n]
                row.append(output_val)
            output_tensor.append(row)
        return output_tensor
