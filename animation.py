from manim import *

class Convolution(Scene):
    def construct(self):
        # Create input signal
        input_signal = [1, 2, 3, 4, 5, 6, 7]
        input_signal_text = MathTex("{[", 1, ",", 2, ",", 3, ",", 4, ",", 5, ",", 6, ",", 7, "]}").scale(0.8).shift(UP*1.5)
        self.play(Write(input_signal_text))

        # Create kernel
        kernel = [0.5, 0.5]
        kernel_text = MathTex("{[", 0.5, ",", 0.5, "]}").scale(0.8).shift(DOWN*0.5)
        self.play(Write(kernel_text))

        # Convolution calculation
        result = [sum(input_signal[i:i+len(kernel)] * kernel) for i in range(len(input_signal)-len(kernel)+1)]
        result_text = MathTex("{[", result[0], ",", result[1], ",", result[2], ",", result[3], ",", result[4], "]}").scale(0.8).shift(DOWN*2.5)
        self.play(Write(result_text))

        # Display convolution process
        for i in range(len(input_signal)-len(kernel)+1):
            input_slice = MathTex("{[", input_signal[i], ",", input_signal[i+1], "]}").scale(0.8).shift(LEFT*2 + DOWN*1)
            self.play(Transform(input_signal_text.copy(), input_slice))
            self.wait(0.5)
            conv_text = MathTex("+").scale(0.8).shift(RIGHT*0.5)
            self.play(Write(conv_text))
            kernel_slice = MathTex("{[", kernel[0], ",", kernel[1], "]}").scale(0.8).shift(RIGHT*1)
            self.play(Transform(kernel_text.copy(), kernel_slice))
            self.wait(0.5)
            result_text[i+1].set_value(result[i+1])
            self.wait(0.5)

        self.wait(2)
