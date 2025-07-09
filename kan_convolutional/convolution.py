# Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape

    h_out = (
        np.floor(
            (n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1))
            / stride[0]
        ).astype(int)
        + 1
    )
    w_out = (
        np.floor(
            (m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1))
            / stride[1]
        ).astype(int)
        + 1
    )
    b = [kernel_side // 2, kernel_side // 2]
    return h_out, w_out, batch_size, n_channels


def multiple_convs_kan_conv2d(
    matrix,  # but as torch tensors. Kernel side asume q el kernel es cuadrado
    kernels,
    kernel_side,
    out_channels,
    stride=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    device="cuda",
) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out, batch_size, n_channels = calc_out_dims(
        matrix, kernel_side, stride, dilation, padding
    )
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out)).to(
        device
    )  # estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold(
        (kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride
    )
    conv_groups = (
        unfold(matrix[:, :, :, :])
        .view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out)
        .transpose(2, 3)
    )  # reshape((batch_size,n_channels,h_out,w_out))
    # for channel in range(n_channels):
    kern_per_out = len(kernels) // out_channels
    # print(len(kernels),out_channels)
    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)

        # Aggregate outputs from each kernel assigned to this output channel
        for k_idx in range(kern_per_out):
            kernel = kernels[c_out * kern_per_out + k_idx]
            conv_result = kernel.conv.forward(
                conv_groups[:, k_idx, :, :].flatten(0, 1)
            )  # Apply kernel with non-linear function
            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum  # Store results in output tensor

    return matrix_out


def add_padding(matrix: np.ndarray, padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix

    return padded_matrix
